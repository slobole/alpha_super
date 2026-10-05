"""Write the English research record docs/research/FUND_PRODUCTS_20261005.md from the study's JSON outputs.

The prose is fixed here; every number in the tables is read from <study>/report/*.json, so the record and the report
page cannot drift apart. Verdict words that depend on a number are computed. Usage: python build_record.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
REP = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "report"
S9 = "S9 incumbent launch"            # internal key (not renamed): the Monthly book, TAA 3x 1N 60 / CORE5 40 (owner decision 2026-10-05)
M2 = "old growth plus"                # internal key of the dropped 65 / 35 book: still in the JSON files, not a product
AGG_APPROVED = "2026-10-05"           # the owner approved the AGGRESSIVE rung on this date
S13 = "S13 old monthly (2026-10-01)"    # the four-pod monthly book of 2026-10-01 (reference)
OLDP = "old monthly plus (2026-10-01)"
S0 = "S0 equal capital (registered default)"
GR1_L = "T2 MR -> BIL (GR1-L)"
POD = {"core5": "CORE5", "dv2_g": "DV2-G", "hpi_g": "HPI-G", "ndx_atr_cap": "NDX ATR cap", "ndx_natr_cap": "NDX NATR cap", "taa3x": "TAA 3x", "taa3x_1n": "TAA 3x 1N",
       "ndx_vxn": "NDX-VXN", "btal_qqq": "BTAL_QQQ"}


def load(name: str) -> dict:
    p = REP / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pct(x, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def f2(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def usd(x) -> str:
    return "n/a" if not x else (f"${round(x / 1e6, 1):g}M" if x >= 1e6 else f"${x / 1e3:.0f}K")


def main() -> int:
    S, B, C, X, D, P, V, M = (load(n) for n in ("study", "battery", "capacity", "exposure", "defensive", "pm_confirm", "versus", "monthly"))
    stale = lambda name: (REP / f"{name}.json").exists() and (REP / f"{name}.json").stat().st_mtime < (REP / "study.json").stat().st_mtime  # noqa: E731
    if stale("pm_confirm"):      # an engine confirmation older than the study belongs to the previous product set
        P = {}
    if stale("monthly"):
        M = {}
    K, HL, ED = S["books"], B["edge_decay"]["headline"], B["edge_decay"]
    dep = B["dependence"]["products"]
    fin = {n: S["products"][n]["final"] for n in ("GR1", "GR2", "GR3")}
    hard = {"GR1": "p20", "GR2": "p25", "GR3": "p30"}
    rung_key = {"GROWTH": "p20", "GROWTH PLUS": "p25", "AGGRESSIVE": "p30"}
    lab = {"GR1": "GR1 Growth", "GR2": "GR2 Growth Plus", "GR3": "GR3 Aggressive"}
    mix = {"GR1": "TAA 3x 40%, momentum capsule 30%, MR capsule 30% (owner decision 2026-10-05)",
           "GR2": "TAA 3x 1N 40%, momentum capsule 30%, MR capsule 30% (owner decision 2026-10-05: the same weights, 1N variant)",
           "GR3": "TAA 3x 1N 60%, momentum capsule 20%, MR capsule 20% (owner decision 2026-10-05)"}
    cap = lambda n, r="worked+blocks": C["books"][n]["routes"][r]["recommended"]  # noqa: E731
    def capx(n: str, r: str = "worked+blocks") -> str:
        """Worked-route capacity, capped by the BTAL 10%-ownership wall when that binds first."""
        e = C["books"][n]
        if r == "worked+blocks" and e.get("btal_wall") and e["btal_wall"] < cap(n, r):
            return usd(e["btal_wall"]) + " (BTAL wall)"
        return usd(cap(n, r)) + (" or more" if e["routes"][r].get("at_grid_top") else "")

    EM = ED.get("edge_margin", {})
    rk = lambda n: rung_key.get(K[n].get("strictest_rung") or "", "p20")  # noqa: E731

    def margin_k(n: str):
        e = EM.get(n)
        return None if not e else ((e.get("by_limit") or {}).get(rk(n)) or {}).get("lowest_passing_k", e.get("lowest_passing_k"))
    sc = ED["scenarios"]
    cons = lambda n: sc["all at 0.75"].get(n) or {}       # conservative case: 3/4 of every engine's excess return, model costs  # noqa: E731
    stress = lambda n: (HL.get(n) or {}).get("floor") or {}  # stress case: half the excess return and +5 bps per side  # noqa: E731
    dead = lambda n: (sc["TAA dead"].get(n) or {}).get("xs")  # noqa: E731
    q3 = lambda n: f"CAGR {pct(K[n]['q']['cagr'])}, excess Sharpe {f2(K[n]['q']['xs'])}, max DD {pct(K[n]['q']['dd'])}"  # noqa: E731
    barw = lambda x: "above" if x >= 0.82 else "at" if x >= 0.79 else "below"  # noqa: E731
    mk = lambda n: "n/a" if margin_k(n) is None else f"{margin_k(n):.2f}"  # noqa: E731
    mw = K[S9]["weights"]
    m_mix = f"TAA 3x 1N {mw.get('taa3x_1n', 0) * 100:.0f} / CORE5 {mw.get('core5', 0) * 100:.0f}"
    rung_m = K[S9].get("strictest_rung")
    thin = (f"The Monthly book passes {rung_m or 'no rung'}" + (f" and not GROWTH ({pct(K[S9]['tails'].get('p20'))} of paths beyond -20%), so it carries more risk than GR1, not less" if rung_m != "GROWTH" else "")
            + f". Its rung holds down to {mk(S9)} of the edge (GR1 {ED['edge_margin']['GR1']['lowest_passing_k']:.2f}); in the conservative case its breach figure at -{rk(S9)[1:]}% is {pct(cons(S9).get(rk(S9)))} against the 15% cap.")
    LEV = (S["margin"].get("GR1 -> GR3") or {}).get("fixed_140") or {}

    def who(share: float, a_: str, b_: str) -> str:
        if share >= 0.79:
            return f"{a_} leads clearly"
        if share >= 0.60:
            return f"a tie by the 80% rule, {a_} ahead on most paths"
        if share > 0.40:
            return "a tie"
        return f"a tie by the 80% rule, {b_} ahead on most paths" if share > 0.21 else f"{b_} leads clearly"

    L: list[str] = []
    a = L.append

    a("# Fund products, final pass: GROWTH from three capsules, one two-pod monthly book, DEFENSIVE verified (2026-10-05)\n")
    a("Research record. Simulation only; not an allocation approval and nothing here is wired to LIVE.\n")
    a("Frozen plan: `scripts/research/fund_products_20261005/SPEC_FROZEN.md` (commit 70f5c22, reviewed by three independent reviewers before the "
      "freeze; its amendment log, written after results, is at its end). Code: the same folder. Outputs: `results/research/portfolio/fund_products_20261005/` (git-ignored). "
      "Report page (Hebrew, English statistics): the \"fund products\" artifact, version 5.2.\n")
    a("Owner decisions of 2026-10-05, all taken after the results and applied here: GR1 = TAA 3x 40 / 30 / 30 (amendment O1); GR2 = the same weights with TAA 3x 1N (O2); "
      "the monthly book is one book of two pods, TAA 3x 1N 60 / CORE5 40 (O3; Monthly Plus was dropped); GR3 = TAA 3x 1N 60 / 20 / 20 and the AGGRESSIVE rung was approved; the backtest is the headline, and the earlier \"planning\" and \"floor\" columns are replaced by one conservative case and one stress case; "
      "the Defensive-to-Growth blend is shown as its own dial.\n")

    a("## Summary\n")
    g1, g2, g3, s9, m2 = K["GR1"], K["GR2"], K["GR3"], K[S9], K[M2]
    rev = next(c for c in S["challenges_reverse"] if c["default"] == S9)
    a("- **DEFENSIVE needs no update.** The stored inputs are byte-identical, the defensive legs' code is unchanged since the sleeve runs, and a re-run of "
      "the A6-d rules from an empty cache reproduced all 1,011 fields with zero differences. Only two side rows depended on the old growth book or on raw "
      "stock mean reversion; their capsule versions are shown beside them and replace nothing.")
    s0 = K[S0]
    v0 = V[f"GR1 | {S0}"]["frames"]["main"]
    c1 = cons("GR1")
    tie0 = "a tie by the 80% rule" if 0.21 < v0["share_xs"] < 0.79 else "ahead" if v0["share_xs"] >= 0.79 else "behind"
    a(f"- **GROWTH is rebuilt from three capsules**, with no weight search: TAA 3x 40%, the momentum capsule (E2 with the sector cap) 30% and the MR "
      f"capsule (DV2-G + HPI-G, BIL parking) 30%. The registered default was equal capital (one third each); on 2026-10-05, after the results, the owner chose the 40 / 30 / 30 dial point, "
      f"a mild tilt to the TAA engine (TAA 3x is the pod that trades live today). Against equal capital ({pct(s0['q']['cagr'])} / {f2(s0['q']['xs'])} / {pct(s0['q']['dd'])}) it is {tie0} on Sharpe "
      f"(higher on {pct(v0['share_xs'], 0)} of paired paths, {barw(v0['share_xs'])} the 80% bar) with a higher CAGR on {pct(v0['share_cagr'], 0)} of paths; the price is more dependence on TAA. "
      f"Backtest (the headline): {q3('GR1')}. Conservative case (every engine keeps three quarters of its historical excess return, at model costs): CAGR {pct(c1.get('cagr'))}, excess Sharpe {f2(c1.get('xs'))}.")
    p3 = S["products"]["GR3"]
    g3_off, g3_step = bool(p3.get("offered", True)), (p3.get("step_over_product_below") or 0.0) * 100
    v21 = (V.get("GR2 | GR1") or {}).get("frames", {}).get("main")
    rung3 = p3.get("strictest_rung")
    hk3 = rung_key.get(rung3 or "", "p30")
    needs_agg = rung3 == "AGGRESSIVE"
    fk = lambda k_: "none" if k_ is None else f"{k_:.2f}"  # noqa: E731
    em_at = lambda n, key: (((ED.get("edge_margin", {}).get(n) or {}).get("by_limit") or {}).get(key) or {}).get("lowest_passing_k")  # noqa: E731
    k2, k3 = em_at("GR2", "p25"), em_at("GR3", hk3)
    agg_rung = (f"GR3 passes only the AGGRESSIVE rung (max DD >= -27%, breach cap 15% at -30%), which is new in this study and was approved by the owner on {AGG_APPROVED}."
                if AGG_APPROVED else "GR3 passes only the new AGGRESSIVE rung (max DD >= -27%, breach cap 15% at -30%), not yet confirmed by the owner.")
    menu = (f"By the plan's pre-registered menu rule (a higher product is offered only if its CAGR is at least 1.5 pp above the product below it) GR3 is NOT offered: its step over GR2 is {g3_step:.1f} pp. "
            "The growth menu is therefore two capsule products (GR1, GR2) plus the monthly book; GR3 stays in the tables as a reference row." if not g3_off
            else f"GR3 adds {g3_step:.1f} pp over GR2 and is offered by the menu rule (1.5 pp). It is the most return on the ladder and a one-engine product." + (" " + agg_rung if needs_agg else ""))
    a(f"- **One ladder for more return.** GR2 is GR1 with the TAA engine's 1N variant at the same 40 / 30 / 30 weights: {q3('GR2')}"
      + (f" (higher excess Sharpe than GR1 on only {pct(v21['share_xs'], 0)} of paired paths, higher CAGR on {pct(v21['share_cagr'], 0)}: it buys return, not Sharpe)" if v21 else "")
      + f". GR3 raises the 1N variant to 60%: {q3('GR3')}. {menu} "
      f"This is more TQQQ, not more alpha: TAA carries {pct(dep['GR1']['risk_share']['TAA'], 0)} / {pct(dep['GR2']['risk_share']['TAA'], 0)} / {pct(dep['GR3']['risk_share']['TAA'], 0)} of the risk in GR1 / GR2 / GR3.")
    ezm = C["books"][S9]["ease"]
    nwm = ", ".join(POD.get(x, x) for x in ezm["not_wired"]) or "nothing"
    a(f"- **One monthly book of two pods** (owner decision; designed after the results from an exploratory grid): Monthly = {m_mix}: {q3(S9)}. "
      f"Monthly trading only; TAA 3x 1N is wired (it has a live route but no pod is running; the pod trading live today, in the owner's account, is TAA 3x) and the only missing wiring is {nwm}. "
      f"It is the book that can run first and that scales. {thin} Monthly Plus (65 / 35) was dropped. The four-pod monthly books of 2026-10-01 are kept as reference rows only.")
    vs = V[f"GR1 | {S9}"]
    vf, vb = vs["frames"], vs["blocks"]
    bB, bC = {n: K[n]["blocks"]["B"] for n in ("GR1", S9)}, {n: K[n]["blocks"]["C"] for n in ("GR1", S9)}
    a(f"- **GR1 against the Monthly book.** {pct(g1['q']['cagr'])} / vol {pct(g1['q']['vol'])} / max DD {pct(g1['q']['dd'])} / excess Sharpe {f2(g1['q']['xs'])} against "
      f"{pct(s9['q']['cagr'])} / {pct(s9['q']['vol'])} / {pct(s9['q']['dd'])} / {f2(s9['q']['xs'])}. GR1 has the higher excess Sharpe on "
      f"{pct(vf['main']['share_xs'], 0)} / {pct(vf['plus5']['share_xs'], 0)} / {pct(vf['plus10']['share_xs'], 0)} of paired bootstrap paths at 0 / +5 / +10 bps per side "
      f"({barw(vf['main']['share_xs'])} the 80% bar at model costs, {barw(vf['plus5']['share_xs'])} it at +5 bps), and by block on {pct(vb['A']['main']['share_xs'], 0)} (2008-2012, proxy era), "
      f"{pct(vb['B']['main']['share_xs'], 0)} (2012-2021), {pct(vb['C']['main']['share_xs'], 0)} (2022-2026) and {pct(vb['RECENT']['main']['share_xs'], 0)} (last three years). "
      f"2012-2021: excess Sharpe {f2(bB['GR1']['xs'])} vs {f2(bB[S9]['xs'])}; since 2022: {f2(bC['GR1']['xs'])} vs {f2(bC[S9]['xs'])} and CAGR {pct(bC['GR1']['cagr'])} vs {pct(bC[S9]['cagr'])} "
      f"({who(vb['C']['main']['share_xs'], 'GR1', 'Monthly')}). GR1 {'passes' if rev['passed'] else 'does not pass'} all seven pre-registered checks against it in the main frame.")
    a(f"- **Why GR1 stays the target: robustness to one engine failing.** A two-pod monthly book has one return engine and a defensive leg, and no backup. "
      f"With the TAA engine's whole excess return removed GR1 keeps an excess Sharpe of {f2(dead('GR1'))}; the Monthly book keeps only {f2(dead(S9))}: that is the honest price of the two-pod book. "
      f"The price of GR1 is daily trading, {C['books']['GR1']['ease']['research_pods']} research pods instead of two, an unmeasured execution cost and a low capacity.")
    a(f"- **Blend dial.** A client between the two base products holds the defensive launch and GR1 in fixed shares; the table below shows 0 / 20 / 40 / 60 / 80 / 100% GR1 with drawdowns "
      f"(daily correlation of the two {f2(S['corr_gr1_defensive']['all'])}; not full diversification, both lean on the TAA engine).")
    a(f"- **Capacity (house model, pre-TCA).** With everything at the open, the only route the backtest models, the capsule products are about {usd(cap('GR1', 'MOO'))} and Monthly {usd(cap(S9, 'MOO'))}. "
      f"The difference is what can be worked: the monthly book's orders are monthly and can be worked over days ({capx(S9)}), the MR capsule's daily stock orders cannot. "
      f"With the MR stocks at the close auction the capsule products reach about {usd(cap('GR1'))}: an upper bound, because the engine does not model that route and the DV2 timing study rejected a same-day close.")
    a("- **Before client money**, in this order: wire CORE5 (it opens the defensive launch and the Monthly book); run the MR capsule in paper to measure slippage "
      "(the gate: at most 4 bps per side over at least 200 stock fills); the momentum capsule is optional, because GR1 with the live NDX rule in its place is almost the same book "
      f"(correlation {f2(S['standins']['GR1 with live NDX rule']['corr_with_gr1'])}). "
      "If the MR gate fails, growth stays the Monthly book and a new frozen plan is needed. No growth book fits today's account.\n")

    a("## What the products are\n")
    a("Plain idea: three return engines with different mechanisms, each selected on the same history, so no weight may depend on an estimated return. "
      "Equal capital needs no estimate and was the registered default; the owner's 40 / 30 / 30 keeps every engine at 30% or more and gives the oldest engine a mild tilt. It is two risk clusters, not three independent bets: "
      "TAA (whose return leg is TQQQ) and the momentum capsule (Nasdaq-100 stocks) are both long Nasdaq when risk is on; the MR capsule buys S&P 500 dips in stress. "
      "The three capsule products are one ladder: GR1 (TAA 3x 40 / 30 / 30), GR2 (the same weights, a pure switch from TAA 3x to TAA 3x 1N), GR3 (TAA 3x 1N 60 / 20 / 20); the satellites always split the rest equally. "
      "The Monthly book holds one return engine (TAA 3x 1N) and one defensive leg (CORE5). Leverage on GR1 (fixed 1.40x) is shown as an alternative route to GR3's level of return for the fund stage; it is not a product.\n")
    a("| Product | Capital weights | Target rung | Strictest rung passed | YAML |\n|---|---|---|---|---|")
    files = {"GR1": "fund_growth.yaml", "GR2": "fund_growth_plus.yaml", "GR3": "fund_growth_aggressive.yaml"}
    for n in fin:
        p = S["products"][n]
        a(f"| {lab[n]}{'' if p.get('offered', True) else ' (NOT offered by the menu rule; reference)'} | {mix[n]} | {p['target_rung']} | {p['strictest_rung']} | `portfolios/{files[n]}` |")
    a(f"| Monthly (runs first, scales) | {m_mix} (owner decision 2026-10-05) | none registered (designed after results) | {K[S9]['strictest_rung']} (holds down to {mk(S9)} of the edge) | `portfolios/fund_growth_monthly.yaml` |")

    if S13 in K:
        a(f"| Reference: Monthly of 2026-10-01 (four pods, not offered) | TAA 3x 1N 38.4 / NDX-VXN 25.6 / CORE5 18 / BTAL_QQQ 18 | GROWTH | {K[S13]['strictest_rung']} | `portfolios/fund_growth_monthly_20261001.yaml` |")
    if OLDP in K:
        a(f"| Reference: Monthly, more return, of 2026-10-01 (four pods, not offered) | TAA 3x 1N 57.4 / NDX-VXN 24.6 / CORE5 9 / BTAL_QQQ 9 | GROWTH PLUS | {K[OLDP]['strictest_rung']} | `portfolios/fund_growth_plus_monthly_20261001.yaml` |")
    a("")
    a("Internal weights: momentum capsule = `ndx_atr_cap` 50 / `ndx_natr_cap` 50; MR capsule = `dv2_g` 50 / `hpi_g` 50 (one shared VIX stress gate, idle cash in BIL). "
      "Book model: pods compound independently and are reset to target weights at the first session of each calendar year (a transfer with no trade and no cost). "
      "Rungs: GROWTH = max DD >= -17% and breach figure at -20% <= 15%; GROWTH PLUS = -22% / -25%; AGGRESSIVE = -27% / -30% (new in this study, not yet formally confirmed by the owner). "
      "A rung passes only if the historical max DD and the bootstrap cap (10-seed mean and worst seed) hold both in the main frame and at +5 bps. No product needed added cash. "
      "In the JSON files the internal keys were not renamed: `S9 incumbent launch` holds the Monthly book, `old growth plus` holds the dropped 65 / 35 book (not a product), and the 2026-10-01 books are "
      f"`{S13}` and `{OLDP}`.\n")

    a("## Results (LONG window 2008-03-04 to 2026-08-19, gross, fair cash)\n")
    a("The backtest is the headline. The engine already charges IBKR commissions and 2.5 bps of slippage per side, and the headline frame credits idle cash, so the cards and tables of the report show the backtest only. "
      "What stays true: no figure is out of sample, every engine was selected on this history, and the product weights and the monthly book were chosen after the results. "
      f"GR1's excess Sharpe was {f2(g1['blocks']['B']['xs'])} in 2012-2021 and {f2(g1['blocks']['C']['xs'])} since 2022. "
      "One conservative case is shown beside the backtest: every engine keeps three quarters of its historical excess return, at model costs. "
      "The earlier \"planning\" column also added 5 bps per side; for TAA and momentum (monthly orders) that double-counted a cost the engine already charges, and it is a real question only for the "
      "MR capsule's daily stock orders, which is what the paper-trading gate measures. The stress case is half the excess return and +5 bps per side. Both are conventions, not forecasts.\n")
    a("| Book | CAGR | Excess Sharpe | Vol | Max DD | Breach figure at its rung | Conservative case: CAGR / excess Sharpe | Stress case: CAGR / excess Sharpe | Net of 2/20 CAGR | CAGR at +5 bps | CAGR, engine cash 0% |\n|---|---|---|---|---|---|---|---|---|---|---|")
    res_rows = [(fin["GR1"], lab["GR1"]), (fin["GR2"], lab["GR2"]), (fin["GR3"], lab["GR3"]), (S9, f"Monthly ({m_mix})"),
                (S13, "Reference: Monthly of 2026-10-01 (four pods)"), (OLDP, "Reference: Monthly, more return, of 2026-10-01 (four pods)"), (S0, "Equal capital, 1/3 each (registered default)"),
                (GR1_L, "GR1 with the MR 30% in BIL")]
    two = lambda r: f"{pct(r.get('cagr'))} / {f2(r.get('xs'))}" if r else "n/a"  # noqa: E731
    for n, name in res_rows:
        if n not in K:
            continue
        b = K[n]
        hk = hard.get(n) or rung_key.get(b.get("strictest_rung") or "", "p20")
        a(f"| {name} | {pct(b['q']['cagr'])} | {f2(b['q']['xs'])} | {pct(b['q']['vol'])} | {pct(b['q']['dd'])} | "
          f"{'at -25%: ' + pct(b['tails']['p25']) + '; ' if hk == 'p30' else ''}at -{hk[1:]}%: {pct(b['tails'][hk])} | {two(cons(n))} | {two(stress(n))} | {pct(b['net']['cagr'])} | "
          f"{pct(b['frames']['s3_plus_5bps']['cagr'])} | {pct(b['frames']['s1_house_cash']['cagr'])} |")
    a("")
    a(thin + "\n")
    em = ED["edge_margin"]
    a("The breach figure is a yardstick on one convention (share of stationary-bootstrap paths with a deeper drawdown; mean block 63, full backtest edge), not the chance of that drawdown. "
      f"For GR1 at -20% it is {pct(g1['tails']['p20'])} by the rule, {pct(B['bootstrap']['blocks']['21']['GR1']['p20'])} at block 21, {pct(B['bootstrap']['blocks']['1']['GR1']['p20'])} with independent days, "
      f"and {pct(cons('GR1').get('p20'))} in the conservative case. GR1 still passes its rung down to {em['GR1']['lowest_passing_k']:.2f} of the edge. "
      f"GR2 holds GROWTH PLUS (-25%) down to {fk(k2)} of the edge (conservative case {pct(cons('GR2').get('p25'))} against the 15% cap). "
      f"GR3 is read on the rung it actually passes, {rung3 or 'none'} (limit -{hk3[1:]}%): its figure is {pct(K['GR3']['tails'].get(hk3))}, it holds down to {fk(k3)} of the edge, "
      f"the conservative case gives {pct(cons('GR3').get(hk3))} against the 15% cap, and the figure is {pct((B['bootstrap']['blocks']['21'].get('GR3') or {}).get(hk3))} at block 21 and "
      f"{pct((B['bootstrap']['blocks']['1'].get('GR3') or {}).get(hk3))} with independent days"
      + (f"; at -25% (the GROWTH PLUS limit, which it does not pass) it is {pct(K['GR3']['tails']['p25'])}, and its max DD is {pct(K['GR3']['q']['dd'])}. " if hk3 != "p25" else ". ")
      + (agg_rung + " " if needs_agg and g3_off else "")
      + f"TAA carries {pct(dep['GR3']['risk_share']['TAA'], 0)} of GR3's risk; the moderate alternative is GR2 (the same 1N variant at 40%, TAA share of risk {pct(dep['GR2']['risk_share']['TAA'], 0)}).\n")

    a("## The blend dial between Defensive and Growth\n")
    BL = S["blends"]
    pts = sorted((int(nm.split()[1]), v) for nm, v in BL.items() if "diluted" in v)
    a("Two base products and one dial: a client between them holds the defensive launch (CORE5 54 / BTAL_QQQ 36 / cash 10) and GR1 in fixed shares, reset annually, and picks the point by the drawdown they can live with. "
      "Descriptive, in sample. The right-hand columns are GR1 diluted with BIL to about the same volatility: the test of whether the defensive book is worth more than cash.\n")
    a("| GR1 share | CAGR | Excess Sharpe | Vol | Max DD | Breach at -10% / -15% / -20% | GFC window | 2022 bear window | About the same volatility with BIL: vol / CAGR / excess Sharpe / max DD |\n|---|---|---|---|---|---|---|---|---|")
    cz = lambda q, k_: pct((q.get("crises") or {}).get(k_))  # noqa: E731
    brow = lambda lab_, q, t, dil: a(f"| {lab_} | {pct(q['cagr'])} | {f2(q['xs'])} | {pct(q['vol'])} | {pct(q['dd'])} | {pct(t.get('p10'))} / {pct(t.get('p15'))} / {pct(t.get('p20'))} | {cz(q, 'gfc')} | {cz(q, 'bear_2022')} | "  # noqa: E731
                                     + (f"GR1 + {dil['cash'] * 100:.0f}% BIL: {pct(dil['q']['vol'])} / {pct(dil['q']['cagr'])} / {f2(dil['q']['xs'])} / {pct(dil['q']['dd'])}" if dil else "n/a") + " |")
    if "defensive launch" in BL:
        brow("0% (defensive launch)", BL["defensive launch"]["q"], BL["defensive launch"]["tails"], None)
    for sh_, v in pts:
        brow(f"{sh_}%", v["q"], v["tails"], v["diluted"])
    brow("100% (GR1)", g1["q"], g1["tails"], None)
    d_c = [(v["q"]["cagr"] - v["diluted"]["q"]["cagr"]) * 100 for _, v in pts]
    d_d = [(v["q"]["dd"] - v["diluted"]["q"]["dd"]) * 100 for _, v in pts]
    d_s = [v["q"]["xs"] - v["diluted"]["q"]["xs"] for _, v in pts]
    ends = max(g1["q"]["xs"], BL.get("defensive launch", {}).get("q", {}).get("xs", 0.0))
    a(f"\nReading: {sum(v['q']['xs'] > ends for _, v in pts)} of the {len(pts)} interior points have a higher excess Sharpe than both ends "
      f"({f2(min(v['q']['xs'] for _, v in pts))} to {f2(max(v['q']['xs'] for _, v in pts))}). Against cash dilution at about the same volatility (the match is approximate: the diluted books carry 1 to 4% more volatility) the blend's CAGR differs by {min(d_c):+.2f} to {max(d_c):+.2f} pp, "
      f"its excess Sharpe by {min(d_s):+.2f} to {max(d_s):+.2f}, and its max DD is shallower by about 0.5 to {max(d_d):.1f} pp on the one historical path; at 20% GR1 cash dilution is slightly better. A small advantage. "
      f"It is not full diversification: GR1 and the defensive launch correlate {f2(S['corr_gr1_defensive']['all'])} on all days and {f2(S['corr_gr1_defensive']['spx_worst5'])} on the worst 5% of S&P 500 days, "
      "because BTAL_QQQ in the defensive book runs on the same Defense First engine as TAA.\n")

    a("## The monthly book: why two pods, and why 60 / 40 (exploratory, run after the results)\n")
    if M.get("ladder"):
        okb = lambda r: bool(r if isinstance(r, bool) else (r or {}).get("pass"))  # noqa: E731
        yn = lambda r: "yes" if okb(r) else "no"  # noqa: E731
        a("The owner asked for the monthly book to be reduced to TAA 3x 1N and CORE5 and chose the 60 / 40 ratio. `monthly.py` ran a grid after the results (2 TAA variants x 4 TAA:CORE5 ratios x momentum 0 / 15 / 30%, "
          "a TAA 3x 1N / CORE5 ladder, two levered rows and four reference books). It is exploratory and not pre-registered; differences between neighbouring rows are small. "
          "The ladder rows show the trade-off behind the ratio: each step adds return and deepens the drawdown.\n")
        a("| Monthly book | CAGR | Excess Sharpe | Vol | Max DD | Breach at -20% / -25% | 2022 bear window | CAGR at +5 bps | GROWTH rung | GROWTH PLUS rung |\n|---|---|---|---|---|---|---|---|---|---|")
        mrow = lambda nm, r: a(f"| {nm} | {pct(r['q']['cagr'])} | {f2(r['q']['xs'])} | {pct(r['q']['vol'])} | {pct(r['q']['dd'])} | {pct(r['tails'].get('p20'))} / {pct(r['tails'].get('p25'))} | "  # noqa: E731
                               f"{pct(r['q']['crises']['bear_2022'])} | {pct(r['plus5']['cagr'])} | {yn(r['rung_growth'])} | {yn(r['rung_growth_plus'])} |")
        for nm, r in M["ladder"].items():
            mrow("TAA 3x " + nm + (" (levered)" if "L" in r else ""), r)
        for nm, r in M["reference"].items():
            mrow("Reference: " + nm, r)
        G = M["grid"]
        withm = [r for r in G.values() if r.get("vs_no_momentum")]
        base_of = lambda r: G[f"{r['taa']} {r['taa_to_core5']:.0%}:{1 - r['taa_to_core5']:.0%} mom 0%"]  # noqa: E731
        sh = [r["vs_no_momentum"]["share_xs"] for r in withm]
        w1n = [r for r in withm if r["taa"] == "taa3x_1n"]
        ddg = [(r["q"]["dd"] - base_of(r)["q"]["dd"]) * 100 for r in withm]
        cr = M.get("corr", {})
        noc, oldm = M["reference"].get("TAA 3x 57 / momentum 43, no CORE5"), M["reference"].get("old monthly (2026-10-01)")
        plain = {k_: v_ for k_, v_ in M["ladder"].items() if "L" not in v_}
        first20 = next((k_ for k_, v_ in plain.items() if v_["q"]["cagr"] >= 0.20), None)
        vm = V.get(f"{S9} | {S13}", {}).get("frames", {}).get("main")
        a("")
        a(f"- BTAL_QQQ is nearly the same engine as TAA 3x 1N (daily correlation {f2(cr.get('btal_qqq | taa3x_1n'))}), so it adds no diversification; CORE5 correlates {f2(cr.get('core5 | taa3x_1n'))} with TAA 3x 1N.")
        a(f"- The momentum capsule replaces NDX-VXN everywhere in the fund, and in a monthly TAA + CORE5 book it does not earn a weight: over the {len(withm)} grid books with momentum, the share of paired paths with a higher "
          f"excess Sharpe than the same book without it is {pct(min(sh), 0)} to {pct(max(sh), 0)}" + (", never 80%" if max(sh) < 0.79 else "") + f" (below 50% in {sum(r['vs_no_momentum']['share_xs'] < 0.5 for r in w1n)} of the {len(w1n)} books with the 1N variant); "
          f"CAGR is equal or lower in {sum(r['q']['cagr'] <= base_of(r)['q']['cagr'] + 0.0005 for r in withm)} of {len(withm)}. What momentum changes is the historical max DD ({min(ddg):+.1f} to {max(ddg):+.1f} pp, positive = shallower), "
          f"and the 2022 window is worse in {sum(r['q']['crises']['bear_2022'] < base_of(r)['q']['crises']['bear_2022'] for r in withm)} of {len(withm)}.")
        solo = K["capsule TAA 3x 1N"]
        a(f"- CORE5 is needed: TAA 3x 1N alone has a max DD of {pct(solo['q']['dd'])} and a breach figure at -20% of {pct(solo['tails']['p20'])} and passes no rung."
          + (f" TAA 3x 57 / momentum 43 without CORE5 also {'fails' if not okb(noc['rung_growth']) else 'passes'} GROWTH, on the breach figure ({pct(noc['tails'].get('p20'))} against the 15% cap; max DD {pct(noc['q']['dd'])})." if noc else ""))
        if oldm:
            a(f"- The new Monthly against the four-pod book of 2026-10-01: {q3(S9)} against CAGR {pct(oldm['q']['cagr'])}, excess Sharpe {f2(oldm['q']['xs'])}, max DD {pct(oldm['q']['dd'])}"
              + (f"; the new book has the higher excess Sharpe on {pct(vm['share_xs'], 0)} of paired paths (below the bar: a tie) and the old book the higher CAGR on {pct(1 - vm['share_cagr'], 0)}" if vm else "")
              + ". Nearly the same book with two pods instead of four.")
        a("- " + thin)
        a("")
    else:
        a("Pending: `monthly.json` (the exploratory monthly-book grid of `monthly.py`) had not been written when this record was built.\n")

    a("## Evidence for the structure\n")
    a("**If one engine stops working** (its whole excess return removed, its volatility kept):\n")
    dfd = "Defense First dead (TAA and BTAL_QQQ)"
    a("| Book | Backtest excess Sharpe | TAA dead | TAA and BTAL_QQQ dead | Momentum dead | MR dead | All at 3/4 (conservative case) | All at 1/2 |\n|---|---|---|---|---|---|---|---|")
    g = lambda s_, n: f2((sc[s_].get(n) or {}).get("xs"))  # noqa: E731
    for n, name in (("GR1", lab["GR1"]), (S0, "Equal capital, 1/3 each"), ("GR2", lab["GR2"]), ("GR3", lab["GR3"]), (S9, "Monthly"),
                    (S13, "Reference: Monthly of 2026-10-01 (four pods)"), ("S4 core + satellites", "TAA 3x 50 / 25 / 25")):
        if n not in K or not sc["TAA dead"].get(n):
            continue
        a(f"| {name} | {f2(K[n]['q']['xs'])} | {g('TAA dead', n)} | {g(dfd, n)} | {g('MOM dead', n)} | {g('MR dead', n)} | {g('all at 0.75', n)} | {g('all at 0.5', n)} |")
    mm = ED["minimax"]
    rival = ("TAA 3x " + " / ".join(f"{float(v) * 100:.0f}" for v in mm["best_rival"].split()[-1].split("/"))) if mm["best_rival"].startswith("nb GR1") else mm["best_rival"]
    a(f"\nThe 40 / 30 / 30 book is {'within 0.02 of' if mm['gr1_is_minimax_within_0.02'] else 'not claimed to be'} minimax: the best worst case among the neighbours and challengers is {f2(mm['best_rival_worst_xs'])} ({rival}) against {f2(mm['gr1_worst_xs'])} for GR1. "
      "The new Monthly holds one TAA pod and no BTAL_QQQ, so for it the column \"TAA and BTAL_QQQ dead\" equals \"TAA dead\"; in the four-pod reference book BTAL_QQQ runs on the same Defense First engine, so there that column removes both.\n")
    a("**Does each capsule earn its slot** (GR1 with one capsule's slot replaced by BIL; share of paired bootstrap paths on which GR1 has the higher excess Sharpe):\n")
    a("| Replaced | CAGR | Excess Sharpe | GR1 higher Sharpe at 0 / +5 / +10 bps | GR1 higher CAGR |\n|---|---|---|---|---|")
    for t, name in (("T1 MOM -> BIL", "Momentum -> BIL"), (GR1_L, "MR -> BIL"), ("T3 TAA -> BIL", "TAA -> BIL"), ("T4 MOM -> QQQ", "Momentum -> QQQ total return")):
        f_ = S["slots"][t]["frames"]
        a(f"| {name} | {pct(K[t]['q']['cagr'])} | {f2(K[t]['q']['xs'])} | {pct(f_['main']['share_xs'], 0)} / {pct(f_['plus5']['share_xs'], 0)} / {pct(f_['plus10']['share_xs'], 0)} | {pct(f_['main']['share_cagr'], 0)} |")
    gate = S["mr_gate_breakeven"]
    mb = 1 - S["slots"]["T1 MOM -> BIL"]["frames"]["main"]["share_xs"]
    t4 = K["T4 MOM -> QQQ"]
    a(f"\nThe MR capsule earns its slot; GR1 and GR1 without MR have the same excess Sharpe at about {gate['xs']['breakeven_bps_per_side']:.0f} bps of extra cost per side "
      f"(the same CAGR at about {gate['cagr']['breakeven_bps_per_side']:.0f} bps). The momentum capsule adds return, not risk-adjusted return: with BIL in its slot the book's excess Sharpe is higher on "
      f"{pct(mb, 0)} of paths ({barw(mb)} the 80% bar) and its CAGR is {(K['GR1']['q']['cagr'] - K['T1 MOM -> BIL']['q']['cagr']) * 100:.1f} pp lower. "
      f"QQQ in the slot: CAGR {pct(t4['q']['cagr'])}, max DD {pct(t4['q']['dd'])}, strictest rung passed {t4.get('strictest_rung') or 'none'}.\n")
    passed = [c["challenger"] for c in S["challenges"] if c["passed"]]
    chd = {c["challenger"]: c for c in S["challenges"]}
    ck_en = {"c0_rung": "rung", "c1_share_ge_80": "80% share", "c2_breach_no_worse": "breach", "c3_exact_xs_higher": "2012+ window", "c4_plus5_xs_not_lower": "+5 bps", "c5_h1_higher": "first half", "c6_h2_higher": "second half"}
    fails = lambda c: ", ".join(v for k_, v in ck_en.items() if not c["checks"][k_]) or "none"  # noqa: E731
    o2 = S["gr2_vs_old_plus"]
    s8, s5, s1c, s0c = chd["S8 GR1 75 / DEF 25"], chd["S5 MR tilt"], chd["S1 no momentum"], chd.get(S0)
    a(f"**Challengers.** {len(S['challenges'])} alternative structures were tested against GR1 with seven checks (rung, 80% paired share on 20,000 paths, breach, 2012+ window, +5 bps, both halves). "
      f"Passed: {', '.join(passed) if passed else 'none'}. GR1 with a defensive quarter (75 / 25): higher excess Sharpe on {pct(s8['share_xs'], 0)} of paths, failed checks: {fails(s8)}; "
      f"its price is {(K['GR1']['q']['cagr'] - K['S8 GR1 75 / DEF 25']['q']['cagr']) * 100:.1f} pp less CAGR. "
      f"More MR and less momentum (50 / 15 / 35; {pct(s5['share_xs'], 0)}) failed: {fails(s5)}. No momentum (TAA 3x 50 / MR 50; {pct(s1c['share_xs'], 0)}) failed: {fails(s1c)}. "
      + (f"Equal capital {'passes' if s0c['passed'] else 'does not pass'} against 40 / 30 / 30 ({pct(s0c['share_xs'], 0)} of paths): two nearly identical books. " if s0c else "")
      + "By the pre-registered rule no challenger replaces a product in this study; a pass would only have meant a forward-tracking candidate, and no pass means no evidence against the default, not confirmation. "
      f"In the other direction GR1 {'passes every check' if rev['passed'] else 'fails (' + fails(rev) + ')'} against the Monthly book in the main frame ({pct(rev['share_xs'], 0)} of paths on excess Sharpe, {pct(rev['share_cagr'], 0)} on CAGR), "
      ".\n")
    a("**Dependence.** Daily correlations: TAA 3x-momentum "
      f"{f2(B['dependence']['full']['TAA 3x']['MOM'])}, TAA 3x-MR {f2(B['dependence']['full']['TAA 3x']['MR'])}, momentum-MR {f2(B['dependence']['full']['MOM']['MR'])}. "
      f"Diversification ratio by half: {f2(dep['GR1']['diversification_ratio']['h1'])} and {f2(dep['GR1']['diversification_ratio']['h2'])} "
      f"(largest correlation move {f2(dep['GR1']['flags']['max_corr_move'])}). The capsules do fall together in the tail: P(B in its worst 5% | A in its worst 5%) is "
      f"{pct(B['dependence']['tail_dependence']['daily']['TAA 3x | MOM'], 0)} / {pct(B['dependence']['tail_dependence']['daily']['TAA 3x | MR'], 0)} / {pct(B['dependence']['tail_dependence']['daily']['MOM | MR'], 0)} "
      f"on daily returns against 5% under independence. Effective number of bets: {f2(dep['GR1']['enb']['full'])} (GR1), {f2(dep['GR2']['enb']['full'])} (GR2), {f2(dep['GR3']['enb']['full'])} (GR3).\n")
    xb = X["books"]
    inp = S.get("inputs") or S.get("meta", {}).get("inputs") or {}
    a("**Exposure look-through (2012-10-02 on; the proxy era has no stored TQQQ series).** TQQQ weight inside the TAA 3x pod: mean "
      f"{pct(X['tqqq_weight']['taa3x']['daily']['mean'], 0)}, 90th percentile {pct(X['tqqq_weight']['taa3x']['daily']['p90'], 0)}, max {pct(X['tqqq_weight']['taa3x']['daily']['max'], 0)} (1N variant: mean {pct(X['tqqq_weight']['taa3x_1n']['daily']['mean'], 0)}). "
      f"Nasdaq-100 look-through notional (3 x TQQQ + momentum stocks) per unit of product NAV, mean / peak: GR1 {xb['GR1']['nasdaq_lookthrough']['mean']:.2f}x / {xb['GR1']['nasdaq_lookthrough']['max']:.2f}x, "
      f"GR2 {xb['GR2']['nasdaq_lookthrough']['mean']:.2f}x / {xb['GR2']['nasdaq_lookthrough']['max']:.2f}x, GR3 {xb['GR3']['nasdaq_lookthrough']['mean']:.2f}x / {xb['GR3']['nasdaq_lookthrough']['max']:.2f}x at target weights. "
      f"With the pod weights the books actually carried between annual resets the peaks were {xb['GR1']['nasdaq_lookthrough_drift']['max']:.2f}x / {xb['GR2']['nasdaq_lookthrough_drift']['max']:.2f}x / "
      f"{xb['GR3']['nasdaq_lookthrough_drift']['max']:.2f}x (in {xb['GR1']['nasdaq_lookthrough_drift']['peak_date'][:7]}); since 2015 at most {xb['GR1']['nasdaq_lookthrough_drift']['max_since_2015']:.2f}x / "
      f"{xb['GR2']['nasdaq_lookthrough_drift']['max_since_2015']:.2f}x / {xb['GR3']['nasdaq_lookthrough_drift']['max_since_2015']:.2f}x. "
      f"A one-day Nasdaq-100 fall of 10% at peak exposure costs about {pct(xb['GR1']['gap_table']['10%']['peak'])} (GR1) and {pct(xb['GR3']['gap_table']['10%']['peak'])} (GR3) by arithmetic "
      "(TQQQ = 3 x the index, momentum and MR stocks beta 1). The sample has no such day and no 2000-02 type Nasdaq bear; the bootstrap cannot produce one. "
      f"On an average day about {pct(xb['GR1']['tbill_like_share']['mean'], 0)} of GR1 is in BIL or cash.\n")
    mg = S["margin"]
    m12, m13 = mg["GR1 -> GR2"]["vol_matched"], mg["GR1 -> GR3"]["vol_matched"]
    m23 = (mg.get("GR2 -> GR3") or {}).get("vol_matched")
    bk2, bk3 = mg["GR1 -> GR2"]["target"]["breach_key"], mg["GR1 -> GR3"]["target"]["breach_key"]

    def lev(m: dict, tgt: dict, hk: str) -> str:
        """Levered book against the ladder product at the pre-registered tolerance (0.3 pp CAGR, 0.5 pp drawdown or breach)."""
        dc, dd_, db = m["q"]["cagr"] - tgt["q"]["cagr"], m["q"]["dd"] - tgt["q"]["dd"], tgt["tails"][hk] - m["tails"][hk]
        better, worse = (dc > 0.003, dd_ > 0.005, db > 0.005), (dc < -0.003, dd_ < -0.005, db < -0.005)
        if any(better) and not any(worse):
            return "the levered book is ahead beyond the tolerance"
        if any(worse) and not any(better):
            return "the ladder product is ahead beyond the tolerance"
        return "mixed" if any(better) else "a tie inside the tolerance"

    a("**Margin instead of the ladder** (debt as a negative-weight pod at DTB3 + 1.5%, reset annually; no margin calls or gaps modelled; L is the ratio of the unlevered volatilities, "
      f"so the realised volatility of the levered book is slightly below the target's; tolerance 0.3 pp CAGR, 0.5 pp drawdown or breach). GR1 x {m12['L']:.2f} against GR2: CAGR {pct(m12['q']['cagr'], 2)} vs {pct(K['GR2']['q']['cagr'], 2)}, "
      f"max DD {pct(m12['q']['dd'])} vs {pct(K['GR2']['q']['dd'])}, breach at -{bk2[1:]}% {pct(m12['tails'][bk2])} vs {pct(K['GR2']['tails'][bk2])}: {lev(m12, K['GR2'], bk2)}; at +5 bps {pct(m12['plus5']['cagr'], 2)} vs "
      f"{pct(K['GR2']['frames']['s3_plus_5bps']['cagr'], 2)}, at a 2.5% spread {pct(m12['spread_250']['cagr'], 2)}. "
      f"GR1 x {m13['L']:.2f} against GR3: {pct(m13['q']['cagr'], 2)} / {pct(m13['q']['dd'])} vs {pct(K['GR3']['q']['cagr'], 2)} / {pct(K['GR3']['q']['dd'])}: {lev(m13, K['GR3'], bk3)}; at +5 bps "
      f"{pct(m13['plus5']['cagr'], 2)} vs {pct(K['GR3']['frames']['s3_plus_5bps']['cagr'], 2)} (leverage also levers the MR capsule's costs). "
      + (f"GR2 x {m23['L']:.2f} against GR3: CAGR {pct(m23['q']['cagr'], 2)} vs {pct(K['GR3']['q']['cagr'], 2)}, max DD {pct(m23['q']['dd'])} vs {pct(K['GR3']['q']['dd'])}: {lev(m23, K['GR3'], bk3)}. " if m23 else "")
      + "No winner is declared: that is a judgement, not a tolerance result. "
      "Leverage raises the peak Nasdaq exposure and divides the capacity by L. With pods in separate Reg-T accounts only the ladder is practical; a fund with one cross-margined account can use either.\n")

    F = load("fees")
    if F.get("books"):
        FB = F["books"]
        sch = [f"{m * 100:g}/{p_ * 100:g}" for m, p_ in F["schedules"]]
        k_ = lambda x: f"${x * 1e3:.0f}K"  # noqa: E731   income a year on $1M of AUM
        a("## Fee income (hedge-fund style schedules)\n")
        a(f"Model: {F.get('model', 'n/a')}. AUM is held at $1M (income is linear in size). No fund expenses (administration, audit, legal) are included; at this size they can take most of the income. "
          "Each cell: average income a year on $1M in the backtest, then in the conservative case (3/4 of the excess return).\n")
        a("| Book | " + " | ".join(c.replace("/", " / ") for c in sch) + " |\n|---|" + "---|" * len(sch))
        for n in ("Defensive launch", "Blend Growth 40 / Defensive 60", "Growth", "Growth Plus", "Aggressive", "Growth x1.40 (leverage)", "Monthly"):
            if n in FB:
                a(f"| {n} | " + " | ".join(f"{k_(FB[n]['backtest'][c]['income_mean'])}, {k_(FB[n]['conservative'][c]['income_mean'])}" for c in sch) + " |")
        fg, fgc = FB["Growth"]["backtest"], FB["Growth"]["conservative"]
        a("\nWhat the client keeps for Growth (CAGR / excess Sharpe after the fee; backtest, then conservative case):\n")
        a("| Schedule | Backtest | Conservative case | Share of the return above T-bills the fee takes (backtest / conservative) |\n|---|---|---|---|")
        for c in sch:
            a(f"| {c.replace('/', ' / ')} | {pct(fg[c]['net_cagr'])} / {f2(fg[c]['net_xs'])} | {pct(fgc[c]['net_cagr'])} / {f2(fgc[c]['net_xs'])} | {pct(fg[c]['share_of_excess_over_bil'], 0)} / {pct(fgc[c]['share_of_excess_over_bil'], 0)} |")
        d = FB.get("Defensive launch", {}).get("backtest", {})
        a("\nRecommendation as written on the page: 1 / 15 for the growth products and 1 / 10 for the defensive product and the blends, with a high-water mark and no hurdle. Reasons: "
          "the fee should take at most about a quarter of the return above T-bills (about a third in the conservative case)"
          + (f" (Growth at 1 / 15: {pct(fg['1/15']['share_of_excess_over_bil'], 0)} and {pct(fgc['1/15']['share_of_excess_over_bil'], 0)})" if "1/15" in fg else "")
          + (f"; at 2 / 20 the Growth client keeps an excess Sharpe of {f2(fg['2/20']['net_xs'])} ({f2(fgc['2/20']['net_xs'])} conservative) against {f2(S['bench']['SPXTR']['xs'])} for the S&P 500 and {f2(S['bench']['QQQ']['xs'])} for QQQ, hard to sell without a live record" if "2/20" in fg else "")
          + (f"; the defensive product cannot carry a high fee (at 2 / 20 the fee takes {pct(d['2/20']['share_of_excess_over_bil'], 0)} of its return above T-bills, at 1 / 10 {pct(d['1/10']['share_of_excess_over_bil'], 0)})" if "2/20" in d and "1/10" in d else "")
          + "; the lever is AUM, not the fee percentage. Two caveats: no fund expenses are included, and a lawyer comes first (who may charge a performance fee, and to whom, depends on licensing and investor type).\n")
        a("The report page no longer shows a header line or net-of-fees rows and columns; fee effects are in its fees section only. The net of 2/20 column in the results table above is kept in this record for reference.\n")

    if LEV:
        nm_ = LEV.get("name", "")
        xl_ = (xb.get(nm_) or {}).get("nasdaq_lookthrough", {}).get("max")
        cz_ = LEV["q"].get("crises") or (K.get(nm_, {}).get("q", {}).get("crises") or {})
        a("**Leverage at a fixed 1.40x as an alternative to GR3** (not a product; fund stage only, one cross-margined account). "
          f"GR1 x 1.40: CAGR {pct(LEV['q']['cagr'], 2)}, excess Sharpe {f2(LEV['q']['xs'])}, max DD {pct(LEV['q']['dd'])}, breach at -25% / -30% {pct(LEV['tails'].get('p25'))} / {pct(LEV['tails'].get('p30'))}, "
          f"2022 window {pct(cz_.get('bear_2022'))}, at +5 bps {pct((LEV.get('plus5') or {}).get('cagr'), 2)}, at a 2.5% financing spread {pct((LEV.get('spread_250') or {}).get('cagr'), 2)}, "
          f"TAA-dead excess Sharpe {f2(dead(nm_))}, peak Nasdaq look-through {'n/a' if not xl_ else f'{xl_:.2f}x'}. "
          f"GR3: CAGR {pct(K['GR3']['q']['cagr'], 2)}, excess Sharpe {f2(K['GR3']['q']['xs'])}, max DD {pct(K['GR3']['q']['dd'])}, breach at -30% {pct(K['GR3']['tails'].get('p30'))}, 2022 window {pct(K['GR3']['q']['crises']['bear_2022'])}, "
          f"at +5 bps {pct(K['GR3']['frames']['s3_plus_5bps']['cagr'], 2)}, TAA-dead {f2(dead('GR3'))}, peak look-through {xb['GR3']['nasdaq_lookthrough']['max']:.2f}x. "
          f"Verdict at the tolerance (0.3 pp CAGR, 0.5 pp drawdown or breach): {lev(LEV, K['GR3'], bk3)}. It keeps the three-engine balance instead of concentrating in TAA; its costs are financing at DTB3 + 1.5%, "
          "no margin calls or gap days in the model, the MR capsule's trading cost levered too, and capacity divided by 1.4.\n")

    a("## Capacity and ease (pre-TCA)\n")
    a("| Book | House model: open (modelled) | close auction | worked + blocks (upper bound) | Per-leg P99 order = 5% of a median day (binding leg) | Largest same-day order, all pods = 5% | BTAL 10%-ownership wall | Pods: in research / in the live build | Not wired | Minimum clean size |\n|---|---|---|---|---|---|---|---|---|---|")
    for n, name in ((fin["GR1"], lab["GR1"]), (fin["GR2"], lab["GR2"]), (fin["GR3"], lab["GR3"]), (S9, "Monthly"), (S13, "Reference: Monthly of 2026-10-01 (four pods)")):
        e = C["books"].get(n)
        if not e:
            continue
        pl = e.get("participation_by_leg")
        a(f"| {name} | {usd(cap(n, 'MOO'))} | {usd(cap(n, 'MOC'))} | {capx(n)} | "
          + (f"{usd(pl['aum_p99_at_5pct']['aum'])} ({pl['aum_p99_at_5pct']['binding_leg'].replace('leg ', '')}, {pl['aum_p99_at_5pct']['binding_symbol']})" if pl else "n/a") + " | "
          f"{usd(e['participation']['aum_max_at_5pct'])} | "
          f"{usd(e['btal_wall'])} | {e['ease']['research_pods']} / {e['ease']['live_pods']} | {', '.join(POD.get(x, x) for x in e['ease']['not_wired']) or 'none'} | {usd(e['ease']['min_clean_size'])} |")
    a("\nThe house model's opening-auction limits are strict (0.05% of volume) and uncalibrated. The worked route works the monthly ETF orders over days (BTAL as blocks) and sends the MR stock orders to the close auction, "
      "which the engine does not model and the DV2 timing study rejected, so for the capsule products it is an upper bound; there the binding leg is the MR capsule (FOX, NWS). "
      "The capsule products are labelled CAPACITY-LIMITED by the pre-registered AUM rule (worked route below $25M); the Monthly book is not. "
      "The MR pods' BIL parking orders are left out of the gates (amendment C1); the BTAL wall is computed on peak BTAL weights (the TAA pods hold BTAL). "
      f"The Monthly book needs only {nwm} wired. The MR capsule's stock turnover is about {C['books']['leg MR']['stock_turnover_x_nav']:.0f} times the pod a year.\n")

    a("## DEFENSIVE\n")
    if D:
        mr_s, mr_g = D["more_return"]["s9_parity"]["pick"], D["more_return"]["gr1"]["pick"]
        gu = D["gated_upgrade"]
        a("Kept as published on 2026-10-01 (launch CORE5 54 / BTAL_QQQ 36 / cash 10 and the other slots). Two rows are shown beside target-stage versions that replace nothing:\n")
        a(f"- More return: the published 2026-10-01 row (growth slice = the four-pod monthly book of that date, kept as published) CAGR {pct(mr_s['q']['cagr'], 2)}, max DD {pct(mr_s['q']['dd'])}; the same scan with GR1 as the growth slice picks "
          f"{mr_g['g'] * 100:.0f}% growth inside the invested part and {mr_g['cash'] * 100:.0f}% cash ({mr_g['g'] * (1 - mr_g['cash']) * 100:.1f}% of capital in growth): CAGR {pct(mr_g['q']['cagr'], 2)}, excess Sharpe {f2(mr_g['q']['xs'])}, max DD {pct(mr_g['q']['dd'])}. "
          "The same code with the old growth book reproduced the stored slot exactly. The GR1 pick is a boundary point: it clears the -5% crisis floor by " + f"{(D['more_return']['gr1']['frontier_launch'][0]['worst'] + 0.05) * 100:.2f} pp; "
          "the next passing points are within 0.2 pp of CAGR, which is noise.")
        c10, h10 = gu["mr_capsule_10"], gu["hpi_vote_10 (stored A6-d row)"]
        a(f"- Gated upgrade: 60/40 at 90% + MR capsule 10% (cash {c10['cash'] * 100:.0f}%) beats the launch on {pct(c10['shares_vs_launch']['main'], 0)} / {pct(c10['shares_vs_launch']['plus5'], 0)} / "
          f"{pct(c10['shares_vs_launch']['plus10'], 0)} of paths at 0 / +5 / +10 bps, against {pct(h10['shares_vs_launch']['main'], 0)} / {pct(h10['shares_vs_launch']['plus5'], 0)} / "
          f"{pct(h10['shares_vs_launch']['plus10'], 0)} for the stored raw-HPI row. It stays behind the MR slippage gate.\n")

    a("## Data verification\n")
    a("- Stored sleeves re-run at HEAD from a clean worktree: taa3x, taa3x_1n, core5, btal_qqq, ndx_vxn, etf_dv2, eom_flow, downshock byte-identical to the stored files; "
      "dv2 and hpi_vote differ by at most 5.5e-5 a day on about 540 days in 2024-2026 (Norgate revision; CAGR -0.0005 / -0.0014 pp), inside the threshold.\n"
      "- The E2 book is bit-identical to the stored 2026-10-04 run; the MR capsule's stored $100K runs reproduce byte for byte; the study's $1M capsule runs are within 0.05 pp of CAGR and correlate 0.9996.\n"
      "- The runs are causal in the end date (two end-date tests). The main checkout's uncommitted edits (another session's live wiring) are backtest-neutral and no stored series came from that tree.\n"
      f"- The predecessor study's four-pod monthly book of 2026-10-01 reproduces exactly in the new pipeline (breach figure at -20% {pct(S['checks']['incumbent_p20'][1], 2)} in both). `a6d.py` reproduces `a6d.json` "
      "(1,011 fields, 0 different) and `report_a6.py` reproduces `report_a6.json` (14,899 fields, 0 different); `a6.py` differs from `a6.json` in 7 crisis-drawdown fields of the old growth rows "
      "(up to 0.36 pp) because `a6.py` was edited after `a6.json` was written on 2026-10-01; no defensive field differs.")
    pm_names = {"GR1": "GR1", "GR2": "GR2", "GR3": "GR3", "M1": "Monthly"}
    done = {c: v for c, v in (P.get("books") or {}).items() if "engine" in v}
    if done:
        missing = [nm for c, nm in pm_names.items() if c not in done]
        a("- Engine confirmation (each product YAML through the PortfolioManager, engine cash 0%, 2013-01-02 on, against the research book model): "
          + "; ".join(f"{pm_names.get(n, n)}: CAGR {pct(v['engine']['cagr'], 2)} vs {pct(v['research_house_cash']['cagr'], 2)}, correlation {v['corr']:.4f}, {'accepted' if v['accepted'] else 'NOT accepted'}"
                      for n, v in done.items())
          + "." + (f" Pending: {', '.join(missing)}." if missing else ""))
    else:
        a("- Engine confirmation (each product YAML through the PortfolioManager against the research book model): pending for the new product set "
          "(the stored `pm_confirm.json` is older than the re-run study and belongs to the previous weights).")
    a("- After the owner's decisions of 2026-10-05 the whole pipeline was re-run and the page and this record were checked by two independent reviewers (numbers of Growth Plus and of the two monthly books of that version rebuilt with separate code; all matched). The later decisions (GR3 60 / 20 / 20, Monthly 60 / 40, leverage 1.40 as an alternative) were re-run and checked by the author only.")
    a("")

    a("## Important to know (direction and size)\n")
    a("| Convention or caveat | Direction | Size |\n|---|---|---|")
    ez1 = C["books"][fin["GR1"]]["ease"]
    rows = [
        ("Engine, Bench and YAML books pay 0% on idle cash; the headline frame credits DTB3 - 0.5% and charges DTB3 + 1.5% on negative cash", "engine conservative",
         f"CAGR gap, percentage points a year: GR1 {pct(K['GR1']['q']['cagr'] - K['GR1']['frames']['s1_house_cash']['cagr'], 2)}; TAA 3x alone {pct(K['capsule TAA 3x']['q']['cagr'] - K['capsule TAA 3x']['frames']['s1_house_cash']['cagr'], 2)} "
         f"(about 2% cash), momentum capsule {pct(K['capsule MOM']['q']['cagr'] - K['capsule MOM']['frames']['s1_house_cash']['cagr'], 2)}, "
         f"MR capsule {pct(K['capsule MR']['q']['cagr'] - K['capsule MR']['frames']['s1_house_cash']['cagr'], 2)}"),
        ("The products are not fully invested", "note", f"about {pct(xb['GR1']['tbill_like_share']['mean'], 0)} of GR1 is BIL or cash on an average day"),
        ("Three cash treatments live side by side: the MR capsule's BIL is a real position (25% withholding, trade costs); TAA and momentum idle cash is credited DTB3 - 0.5% without cost; "
         "the cash leg of a book is BIL total return", "conservative for MR",
         f"about 0.6 pp of capsule CAGR; GR1 {pct(K['GR1']['frames']['s9_mr_fair_cash']['cagr'] - K['GR1']['q']['cagr'], 2)} under the symmetric treatment"),
        ("TAA before 2012-10-02 is a synthetic TQQQ / BTAL proxy (a quarter of the sample, the only 2008)", "unknown",
         f"GR1 on 2012+ only: CAGR {pct(K['GR1']['frames']['s6_exact']['cagr'])}, excess Sharpe {f2(K['GR1']['frames']['s6_exact']['xs'])}"),
        ("TAA commissions on split-adjusted TQQQ shares", "conservative", "about 0.3 pp a year per TAA pod on average 2012-2026 (0.38 pp in the 1N variant), near zero now"),
        ("Negative cash is not financed in the engine", "optimistic, small",
         f"MR pods {inp['dv2_g']['negative_cash_day_count_int']} and {inp['hpi_g']['negative_cash_day_count_int']} days, to {pct(inp['dv2_g']['minimum_cash_nav_weight_float'])} of the pod; "
         f"momentum pods about {inp['ndx_atr_cap']['negative_cash_day_count_int']} days, to {pct(inp['ndx_atr_cap']['minimum_cash_nav_weight_float'])}; TAA pods about 1,500 days (2012 on), to -1.8%; charged DTB3 + 1.5% in the headline frame"),
        ("Two of the three engines are long Nasdaq when risk is on (TAA through TQQQ, momentum through Nasdaq-100 stocks); no 2000-02 type bear in the sample", "optimistic",
         f"peak look-through {xb['GR1']['nasdaq_lookthrough']['max']:.2f}x (GR1), {xb['GR3']['nasdaq_lookthrough']['max']:.2f}x (GR3) at target weights; "
         f"{xb['GR1']['nasdaq_lookthrough_drift']['max']:.2f}x and {xb['GR3']['nasdaq_lookthrough_drift']['max']:.2f}x as carried; a 10% one-day Nasdaq fall at peak exposure costs "
         f"{pct(xb['GR1']['gap_table']['10%']['peak'])} and {pct(xb['GR3']['gap_table']['10%']['peak'])}"),
        ("Every engine was selected on this history; nothing is out of sample (MR: about 110 variants; momentum: about 45 trials plus a 151-configuration grid; the products: 3 of 14 books seen before the freeze)",
         "optimistic", f"the backtest is the headline (GR1 CAGR {pct(K['GR1']['q']['cagr'])}, excess Sharpe {f2(K['GR1']['q']['xs'])}); conservative case, 3/4 of the excess return at model costs: {pct(c1.get('cagr'))} / {f2(c1.get('xs'))}; "
                       f"GR1's excess Sharpe was {f2(g1['blocks']['B']['xs'])} in 2012-2021 and {f2(g1['blocks']['C']['xs'])} since 2022"),
        ("The product weights were chosen by the owner after the results, from the pre-registered dial points (GR1 and GR2 at 40 / 30 / 30; the registered default was equal capital; the 40 / 30 / 30 point was not in the grid seen before the freeze)", "optimistic",
         f"GR1 against equal capital: {tie0} on Sharpe ({pct(v0['share_xs'], 0)} of paired paths)"),
        (f"The Monthly book ({m_mix}) was designed after the results from an exploratory grid that was not pre-registered; the owner chose the ratio", "optimistic",
         f"differences between neighbouring grid rows are small; one return engine: with TAA dead the Monthly book keeps an excess Sharpe of {f2(dead(S9))} (GR1 {f2(dead('GR1'))})"),
        ("Breach figures are a block-63, full-edge convention", "optimistic", f"GR1 at -20%: {pct(K['GR1']['tails']['p20'])} by the rule, {pct(cons('GR1').get('p20'))} in the conservative case"),
        ("MR capsule cost sensitivity; no live fills; the slippage gate (4 bps per side over 200 fills) is unmet; 2020-21 are about a third of its log-wealth; "
         f"stock turnover about {C['books']['leg MR']['stock_turnover_x_nav']:.0f} times the pod a year", "optimistic until measured",
         f"capsule alone {pct(K['capsule MR']['q']['cagr'])} -> {pct(K['capsule MR']['frames']['s3_plus_5bps']['cagr'])} at +5 bps; GR1 {pct(K['GR1']['q']['cagr'])} -> {pct(K['GR1']['frames']['s3_plus_5bps']['cagr'])}"),
        ("Momentum capsule: today's GICS labels; selection unproven against QQQ at the same exposure; in a drawdown of about 18% from its June 2026 peak (at 2026-10-02)", "unknown",
         f"in the slot test BIL in its place has the higher excess Sharpe on {pct(mb, 0)} of paths "
         f"and {(K['GR1']['q']['cagr'] - K['T1 MOM -> BIL']['q']['cagr']) * 100:.1f} pp less CAGR: it adds return, not risk-adjusted return"),
        ("Annual reset is a cost-free transfer on one date", "optimistic, small", f"GR1 CAGR over the twelve start months {pct(B['reset']['GR1']['start_month_spread']['cagr']['min'])} to {pct(B['reset']['GR1']['start_month_spread']['cagr']['max'])}"),
        ("Capacity is pre-TCA and route-dependent", "unknown",
         f"GR1 {usd(cap('GR1', 'MOO'))} at the open (modelled), up to {usd(cap('GR1'))} if the MR stocks trade at the close (upper bound, not modelled); binding legs: TAA / BTAL on one-day routes, "
         f"MR stocks (FOX, NWS) on the worked route; Monthly {usd(cap(S9, 'MOO'))} at the open, {capx(S9)} worked"),
        ("The Monthly book's rung and its margin", "optimistic", thin),
        ("Leverage on GR1 (fixed 1.40x) is an alternative, not a product", "optimistic", "financing at DTB3 + 1.5%, no margin calls or gap days modelled, leverage also levers the MR capsule's trading cost and divides capacity by 1.4; not available while each pod sits in its own Reg-T account"),
        ("Minimum clean size", "note", f"GR1 about {usd(ez1['min_clean_size'])}, Monthly about {usd(ezm['min_clean_size'])}"),
        ("Wiring", "note", "wired: TAA 3x, TAA 3x 1N, BTAL_QQQ, NDX-VXN (of these only TAA 3x has a pod trading live today; TAA 3x 1N has a live route but is not running); not wired: CORE5 (PM_READY) and the four momentum and MR capsule pods (PM_READY, no live route at 5c0d48d; "
         f"the MR capsule needs a new order shape and two margin accounts); the Monthly book needs only {nwm}"),
        ("Margin rows", "optimistic", "DTB3 + 1.5%, no margin calls, no gap; volatility match is approximate; capacity divides by L"),
        ("Gross against net of 2/20", "note", f"GR1 {pct(K['GR1']['q']['cagr'])} gross, {pct(K['GR1']['net']['cagr'])} net; Monthly {pct(s9['q']['cagr'])} gross, {pct(s9['net']['cagr'])} net; no fund expenses"),
        ("Sharpe basis", "note", f"GR1 excess Sharpe {f2(K['GR1']['q']['xs'])} daily, {f2(K['GR1']['q']['xs_monthly'])} monthly, {f2(K['GR1']['q']['sharpe0'])} at a zero rate"),
        ("The weeks after the window are not out of sample", "note",
         f"2026-08-20 to 2026-10-02: GR1 {pct(B['after_window']['books']['GR1']['after_window'])}, Monthly {pct(B['after_window']['books'][S9]['after_window'])}"),
        ("The engine BIL pod is not BIL total return", "note", "1.08% vs 1.61% a year from 2012 (withholding, no reinvestment)"),
    ]
    for r in rows:
        a(f"| {r[0]} | {r[1]} | {r[2]} |")
    a("")

    a("## Owner decisions\n")
    a("Decided:\n")
    a("1. DECIDED 2026-10-05: GR1 = TAA 3x 40 / momentum 30 / MR 30 is the growth product to build toward (`fund_growth.yaml`). The reason is engine diversification, not a proven Sharpe gain.\n"
      "2. DECIDED 2026-10-05: GR2 (Growth Plus) = the same 40 / 30 / 30 with the TAA 3x 1N variant (`fund_growth_plus.yaml`).\n"
      "2b. DECIDED 2026-10-05: GR3 (Aggressive) = TAA 3x 1N 60 / momentum 20 / MR 20 (`fund_growth_aggressive.yaml`; it was 50 / 25 / 25). The ladder is TAA 3x 40, TAA 3x 1N 40, TAA 3x 1N 60.\n"
      f"3. DECIDED 2026-10-05: one monthly book of two pods: Monthly = {m_mix} (`fund_growth_monthly.yaml`). Monthly Plus was dropped (`fund_growth_monthly_plus.yaml` deleted). "
      "It runs first and is the scalable alternative; if the MR slippage gate fails, growth stays the Monthly book. The four-pod books of 2026-10-01 are reference rows.\n"
      f"3b. DECIDED {AGG_APPROVED}: the AGGRESSIVE rung (max DD >= -27%, breach cap 15% at -30%) is approved; GR3 passes only that rung.\n"
      "3c. Leverage on GR1 at a fixed 1.40x is shown as an alternative route to GR3's level of return for the fund stage (one cross-margined account); it is not a product on the menu and no decision is asked.\n"
      "4. DECIDED 2026-10-05: the backtest is the headline; one conservative case (3/4 of the excess return, model costs) and one stress case (1/2, +5 bps) are shown in the \"how much to believe\" section only.\n")
    a("Open:\n")
    a(("" if g3_off else f"1. GR3 is not offered by the menu rule (step over GR2 {g3_step:.1f} pp, below 1.5 pp).\n") +
      "2. Run the MR capsule in paper or small live and measure slippage: the gate is at most 4 bps per side over at least 200 stock fills.\n"
      "3. Wiring order: CORE5 first (it opens the defensive launch and the Monthly book), then the MR capsule (in progress; paper first), then, optionally, one momentum strategy that averages the two books; "
      "the live NDX rule is an acceptable stand-in for the momentum capsule.\n"
      f"4. MR execution route: at the open GR1 is a product of about {usd(cap('GR1', 'MOO'))}; a close-auction route would lift it to about {usd(cap('GR1'))} at most and has no supporting evidence yet.\n"
      "5. Commit and merge of this study (it lives on a worktree branch).\n")
    a("## Independent review\n")
    a("Version 5.2 (this record): after the owner's decisions of 2026-10-05 (Growth 40 / 30 / 30, Growth Plus 40 / 30 / 30 with the 1N variant, two-pod monthly books, the backtest as the headline; later the same day: GR3 = 60 / 20 / 20, one Monthly book at 60 / 40, the AGGRESSIVE rung approved) the study was re-run and the page and record "
      "then checked by two independent reviewers: all numbers of Growth Plus and of the two monthly books of that version were rebuilt with separate code and matched; the one gap they found, the monthly books' thin rung margin, "
      "is now stated on the page and here, and about twenty wording and labelling items were corrected.\n")
    a("History. The three rounds below were run on the equal-capital version; after the owner's 40 / 30 / 30 decision for GR1 the whole study was re-run, the engine confirmation repeated and the updated page and record reviewed by two more independent reviewers (the GR1 numbers were rebuilt with separate code; leftover equal-capital wording was corrected). "
      "The equal-capital outputs are kept in `results/research/portfolio/fund_products_20261005/report_equal_capital_snapshot/`. Three rounds by independent agents with their own code. (1) Before the freeze, three reviewers blocked the first draft of the plan (incomplete disclosure of what had been seen, the MR gate dropped, "
      "undefined rules); it was rewritten and then frozen. (2) Eight data audits: every stored sleeve re-run at HEAD, input mapping, freshness, causality. (3) Five reviewers of the results: two rebuilt every "
      "headline number from the raw path and transaction files (272 book comparisons, largest difference below 1e-6); a timing and look-ahead audit found none (the VIX gate is aligned to the prior close, "
      "100% of DV2-G buy fills fall on gate-open sessions; the new sleeves re-run to 2019-12-31 match the full runs to 1e-11). No headline number was wrong. What changed after the review, all toward less optimism "
      "(figures of the equal-capital version; \"monthly book\" there means the four-pod book of 2026-10-01): the GR1-over-monthly share is shown at +5 and +10 bps and by block and the summary was rewritten; capacity was corrected (close-auction capacity $10M to $5M, per-leg participation $34.5M to $3.0M); "
      "the carried Nasdaq peak is shown beside the target-weight peak; GR3 is read at -25% as well as -30%; the four-pod monthly book's dead-engine figure includes BTAL_QQQ (0.31); the margin reading, the crisis-window labels "
      "and the amendment log were fixed. The timing reviewer's own estimate of a halved Nasdaq premium at time-average exposure (equal-capital GR1 11.9% / 0.85; not recomputed for 40 / 30 / 30) is quoted beside the study's regression-beta row; "
      f"it is harsher than the conservative case used here (equal capital at 3/4 of the excess return: {pct(cons(S0).get('cagr'))} / {f2(cons(S0).get('xs'))}).\n")
    a("## Computed, in the JSON files, not on the report page\n")
    a("Correlation matrices for blocks B and C, each half and the five crises (`battery.json` dependence); rolling-correlation P10 and P90; the ten worst 21-session windows of GR2 and GR3; "
      "breach figures at -10, -15, -17, -22, -27 and -35% with worst-seed values (`study.json` tails; GR1 at -17%: "
      f"{pct(K['GR1']['tails']['p17'])}, worst seed {pct(K['GR1']['tails']['p17_max'])}); gap tables of the dial-map and margin books (`exposure.json`); gross CAGR percentiles (`battery.json` bootstrap); "
      "the full monthly-book grid with momentum at 15% (`monthly.json`); the old \"planning\" rows (3/4 of the excess return and +5 bps) are still in `battery.json` edge_decay.headline.\n")
    a("## Files\n")
    a("Capsule products: `portfolios/fund_growth.yaml` (GR1), `fund_growth_plus.yaml` (GR2), `fund_growth_aggressive.yaml` (GR3). Monthly book: `portfolios/fund_growth_monthly.yaml` (Monthly, TAA 3x 1N 60 / CORE5 40); `fund_growth_monthly_plus.yaml` was deleted with the Monthly Plus book. "
      "The four-pod books of 2026-10-01 are kept under dated names: `portfolios/fund_growth_monthly_20261001.yaml` and `portfolios/fund_growth_plus_monthly_20261001.yaml`. "
      "Deleted: `fund_growth_mr.yaml` (its HPI-RSI pod was demoted). This work lives on a worktree branch; nothing was changed in the main checkout or pushed.\n")
    a("## Forward review triggers (fixed in the plan)\n")
    a("A trigger opens a review; it is not an automatic exit: a capsule's live or paper drawdown beyond its planning maximum (momentum -30%, MR -21%, TAA -26%); "
      "MR slippage above 4 bps per side over 200 fills; a capsule behind BIL over a rolling three years.\n")
    a("## Reproduce\n")
    a("From the worktree root, with `PYTHONDONTWRITEBYTECODE=1`: `python scripts/research/fund_products_20261005/build_sources.py` (six engine runs), then in that folder "
      "`study.py`, `battery.py`, `exposure.py`, `capacity.py`, `defensive.py`, `versus.py`, `pm_confirm.py write`, the PortfolioManager runs, `pm_confirm.py compare`, `monthly.py`, `build_report.py`, `build_record.py`. "
      "The study imports the shelf-rebuild library from the main checkout (`scripts/research/shelf_rebuild_20260929/lib.py`, untracked there; its hash is in the study ledger).\n")
    out = WT / "docs" / "research" / "FUND_PRODUCTS_20261005.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)

    # The caveat record of the repository: one section for the fund product books, replaced on every build.
    cav = WT / "docs" / "strategies" / "book-strategy-caveats.md"
    text = cav.read_text(encoding="utf-8")
    start, end = "<!-- fund-products-20261005:start -->", "<!-- fund-products-20261005:end -->"
    sec = [start, "## Fund product books (growth capsules and monthly books) — `fund_growth.yaml`, `fund_growth_plus.yaml`, `fund_growth_aggressive.yaml`, `fund_growth_monthly.yaml`\n",
           "Books of the 2026-10-05 fund-products study ([record](../research/FUND_PRODUCTS_20261005.md)): TAA, the momentum capsule and the MR capsule at fixed capital weights "
           "(GR1 TAA 3x 40 / 30 / 30, GR2 the same with TAA 3x 1N, GR3 TAA 3x 1N 60 / 20 / 20), and one monthly book of two pods (TAA 3x 1N 60 / CORE5 40). "
           "Each pod's own caveats above still apply; these rows are about the books.\n",
           "| Caveat | Direction | Size | Evidence |\n|---|---|---|---|"]
    for r in rows:
        sec.append(f"| {r[0]} | {r[1][0].upper() + r[1][1:]} | {r[2]} | Fund products record 2026-10-05 |")
    sec.append(end)
    block = "\n".join(sec) + "\n"
    if start in text:
        text = text[:text.index(start)] + block + text[text.index(end) + len(end):].lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    if "docs/research/FUND_PRODUCTS_20261005.md" not in text:
        text = text.replace("  - docs/research/MOMENTUM_DECISION_20261004.md\n", "  - docs/research/MOMENTUM_DECISION_20261004.md\n  - docs/research/FUND_PRODUCTS_20261005.md\n")
    cav.write_text(text, encoding="utf-8")
    print("updated", cav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
