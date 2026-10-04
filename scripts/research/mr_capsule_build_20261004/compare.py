"""MR capsule build check: the real-engine pods vs the research record (SPEC: docs/research/MR_CAPSULE_20261003.md).

1. Stock-trade parity: HPI-G engine vs the research engine run (check_b gated); DV2-G engine vs the research replica.
2. Returns: engine pods vs the research pods with the final parking (SPMO only while the gate is closed, weekly
   re-weight, T-bill rate for the rest), on 2015-11-02..2026-09-24 (real SPMO) and 2004..2026-09-24.
3. Parking behaviour and costs from the engine trades (SPMO zero after gate-open decisions, order counts, commissions).
4. Cash-only engine runs (parking disabled): HPI-G NAV vs the research engine run (should be identical) and the
   parking effect (parked minus cash-only) in the engine vs the research record.
5. Deflated Sharpe of the engine capsule (excess of the T-bill rate), N = 110 trials counted as independent
   (about 100 gate variants plus about 10 parking forks), trial variance unknown (null sampling variance).
6. BIL policy: the weekly sweep (B2) vs re-targeting BIL on every close (run2_p1_guard, DV2 only).
7. Is SPMO worth it vs BIL / T-bills? Capsule and book (TAA 0.5 + NDX 0.25 + capsule 0.25), research record
   (T-bills both vs SPMO both + rule) and engine (BIL only vs the spec), by sub-period.
Windows: 2004 on, 2007-06 on (BIL exists), 2015-11 on (real SPMO), 2018-02 on (SPMO guard B1 inactive).
Run after run_engine.py {dv2,hpi} {parked,cash}. Writes results/research/mr_capsule_build_20261004/compare.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "mr_capsule_20261003"))
import spmo_rebalance as sr  # noqa: E402  (research modules: ev, cp, fp, rs, npc)

from alpha.stats.psr_dsr import deflated_sharpe_ratio  # noqa: E402
from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser, stress_gate_open_ser  # noqa: E402

ev, cp, fp, rs, npc = sr.ev, sr.cp, sr.fp, sr.rs, sr.npc
OUT = REPO / "results/research/mr_capsule_build_20261004"
CHECKS = REPO / "results/research/mr_gate_final_checks_20261003"
PARK = ("SPMO", "BIL")
END = "2026-09-24"
SPMO_START = "2015-11-02"
TRIAL_COUNT_FLOAT = 110.0


def load_engine(pod: str, mode: str = "parked", folder: Path = OUT):
    nav = pd.read_csv(folder / f"{pod}_{mode}_nav.csv", index_col=0, parse_dates=True).astype(float)
    tx = pd.read_csv(folder / f"{pod}_{mode}_transactions.csv", parse_dates=["bar"])
    diag = json.loads((folder / f"{pod}_{mode}_diagnostics.json").read_text(encoding="utf-8"))
    return nav, tx, diag


def event_set(asset, bar, side) -> set:
    return set(zip(pd.Series(asset).astype(str), pd.to_datetime(pd.Series(bar)).dt.normalize(), pd.Series(side).astype(int)))


def parity(engine_set: set, research_set: set, lo, hi) -> dict:
    a = {e for e in engine_set if lo <= e[1] <= hi}
    b = {e for e in research_set if lo <= e[1] <= hi}
    return {"engine_events": len(a), "research_events": len(b), "common": len(a & b),
            "engine_only": len(a - b), "research_only": len(b - a), "jaccard": len(a & b) / max(len(a | b), 1),
            "engine_only_sample": sorted((x[0], str(x[1].date()), x[2]) for x in a - b)[:8],
            "research_only_sample": sorted((x[0], str(x[1].date()), x[2]) for x in b - a)[:8]}


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    s = fp.stats(r)
    years = len(r) / 252.0
    return {"cagr": float((1 + r).prod() ** (1 / years) - 1), "sharpe": float(s["sharpe"]), "max_dd": float(s["max_dd"]),
            "wealth_100k": float(100_000 * (1 + r).prod())}


def research_pods(idx: pd.DatetimeIndex) -> dict:
    """Final-spec research pods (as spmo_gateoff.sleeve_gateoff, 'both SPMO'), plus the T-bill-only parking."""
    comp = pd.read_parquet(cp.OUT / "components.parquet").reindex(idx)
    rate = rs.cash_rate(idx)
    gate = pd.Series(cp.gate_on(idx), index=idx)
    spmo_raw = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(idx)
    spmo = spmo_raw.fillna(0.0)
    rv = spmo_raw.rolling(20).std() * np.sqrt(252)
    # before SPMO's first return the sleeve is all T-bills (a zero-return stand-in would get weight 1)
    target = (0.08 / rv).clip(upper=1).where(spmo_raw.notna()).fillna(0.0).to_numpy()
    wk = idx.isocalendar().week.to_numpy()
    wk_end = np.r_[wk[1:] != wk[:-1], True]
    sv, bv, g = spmo.to_numpy(), rate.fillna(0.0).to_numpy(), gate.to_numpy()
    w, held, ret = 0.0, np.zeros(len(idx)), np.zeros(len(idx))
    for t in range(len(idx)):
        held[t] = w
        p = w * sv[t] + (1 - w) * bv[t]
        ret[t] = p
        w = w * (1 + sv[t]) / (1 + p)
        if g[t]:
            w = 0.0
        elif wk_end[t] or (t > 0 and g[t - 1]):
            w = target[t]
    pr, hw = pd.Series(ret, index=idx), pd.Series(held, index=idx)
    out = {}
    for name in ("DV2-G", "HPI-G"):
        cw = comp[f"{name}|engine|cw"]
        e = cw * hw
        out[name] = comp[f"{name}|engine|base"] + cw * pr - sr.COST * e.diff().abs().fillna(0.0)
        out[name + "|tbill"] = comp[f"{name}|engine|base"] + cw * rate
    return out


def spmo_vs_tbills(eng: dict, eng_bil: dict, res: dict) -> dict:
    """Capsule and book metrics for T-bill/BIL-only parking vs the SPMO spec, research and engine, by sub-period."""
    tbc = sr.tbc
    taa, ndx = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    variant_dict = {
        "research_tbills": {"DV2": res["DV2-G|tbill"], "HPI": res["HPI-G|tbill"]},
        "research_spmo": {"DV2": res["DV2-G"], "HPI": res["HPI-G"]},
        "engine_bil": {"DV2": eng_bil["DV2-G"], "HPI": eng_bil["HPI-G"]},
        "engine_spmo": {"DV2": eng["DV2-G"], "HPI": eng["HPI-G"]},
    }
    out = {}
    for start, end in (("2008-03-04", "2026-08-19"), ("2015-11-02", "2018-01-31"), ("2018-02-01", "2026-08-19")):
        block = {}
        for name, part_dict in variant_dict.items():
            cap = ev.capsule(part_dict, {"DV2": .5, "HPI": .5}, start=start, end=end)
            book = tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": cap}, npc.CANDIDATE_WEIGHT_DICT, start, end)
            block[name] = {"capsule": stats(cap), "book": stats(book)}
        out[f"{start}_{end}"] = block
    return out


def parking_behaviour(nav: pd.DataFrame, tx: pd.DataFrame, gate_ser: pd.Series) -> dict:
    cal = nav.index
    pos = tx.pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum").reindex(cal).fillna(0.0).cumsum()
    prior = pd.Series(cal[:-1], index=cal[1:])
    gate_at_decision = gate_ser.reindex(pd.DatetimeIndex(prior.values), method="ffill").fillna(False).to_numpy().astype(bool)
    spmo_pos = pos["SPMO"].reindex(prior.index).to_numpy() if "SPMO" in pos else np.zeros(len(prior))
    park_tx = tx[tx["asset"].isin(PARK)]
    years = len(cal) / 252.0
    res = {
        "spmo_held_after_gate_open_decision_sessions": int((spmo_pos[gate_at_decision] > 0).sum()),
        "orders_per_year": {a: float((park_tx["asset"] == a).sum() / years) for a in PARK},
        "stock_orders_per_year": float((~tx["asset"].isin(PARK)).sum() / years),
        "sessions_with_short_position": int((pos < -1e-9).any(axis=1).sum()),
        "transaction_columns": list(tx.columns),
    }
    comm_col = next((c for c in ("commission", "commission_float", "fees") if c in tx.columns), None)
    if comm_col:
        avg_nav = float(nav["total_value"].mean())
        res["parking_commission_total"] = float(park_tx[comm_col].sum())
        res["parking_commission_pp_per_year_at_avg_nav"] = float(park_tx[comm_col].sum() / years / avg_nav * 100)
        orders_per_year = len(park_tx) / years
        res["min_commission_friction_pp_per_year"] = {f"{k}k_pod": float(orders_per_year * 1.0 / (k * 1000) * 100) for k in (6, 15, 50)}
    return res


def main() -> None:
    rep: dict = {}
    vix_gate = stress_gate_open_ser(load_vix_close_ser())
    engine = {p: load_engine(p) for p in ("dv2", "hpi")}
    cash = {p: load_engine(p, "cash") for p in ("dv2", "hpi")}
    bil = {p: load_engine(p, "bil") for p in ("dv2", "hpi")}
    lo, hi = pd.Timestamp("2004-01-01"), pd.Timestamp(END)
    # ---- 1. stock-trade parity
    hold = pd.read_parquet(cp.OUT / "holdings.parquet")
    dv2_rep = hold[hold["variant"] == "DV2-G"]
    exited = dv2_rep[dv2_rep["exit_date"].notna()]
    dv2_rep_set = event_set(dv2_rep["asset"], dv2_rep["entry_date"], np.ones(len(dv2_rep))) | event_set(
        exited["asset"], exited["exit_date"], -np.ones(len(exited)))
    hpi_rep_nav = pd.read_csv(CHECKS / "hpi_gated_nav.csv", index_col=0, parse_dates=True).astype(float)
    hpi_rep_tx = pd.read_csv(CHECKS / "hpi_gated_transactions.csv", parse_dates=["bar"])
    hpi_rep_set = event_set(hpi_rep_tx["asset"], hpi_rep_tx["bar"], np.sign(hpi_rep_tx["amount"]))
    hpi_hi = min(hi, hpi_rep_tx["bar"].max())
    for mode, runs in (("parked", engine), ("cash", cash)):
        for pod, research_set, top in (("dv2", dv2_rep_set, hi), ("hpi", hpi_rep_set, hpi_hi)):
            stock = runs[pod][1][~runs[pod][1]["asset"].isin(PARK)]
            rep[f"{pod}_{mode}_trade_parity"] = parity(event_set(stock["asset"], stock["bar"], np.sign(stock["amount"])), research_set, lo, top)
    common = cash["hpi"][0].index.intersection(hpi_rep_nav.index)
    rel = (cash["hpi"][0]["total_value"].reindex(common) / hpi_rep_nav["total_value"].reindex(common) - 1).abs()
    rep["hpi_cash_vs_research_engine_nav"] = {"sessions": int(len(common)), "max_abs_rel_diff": float(rel.max()),
                                              "last_date": str(common[-1].date())}
    # ---- 2. returns (parked engine vs research final spec) and the parking effect
    idx = engine["dv2"][0].index.intersection(engine["hpi"][0].index)
    pct = lambda nav: nav["total_value"].pct_change().reindex(idx)  # noqa: E731
    eng = {"DV2-G": pct(engine["dv2"][0]), "HPI-G": pct(engine["hpi"][0])}
    eng_cash = {"DV2-G": pct(cash["dv2"][0]), "HPI-G": pct(cash["hpi"][0])}
    eng_bil = {"DV2-G": pct(bil["dv2"][0]), "HPI-G": pct(bil["hpi"][0])}
    res = research_pods(idx)
    comp = pd.read_parquet(cp.OUT / "components.parquet").reindex(idx)
    for start, label in (("2004-01-05", "2004_on"), ("2007-06-01", "2007-06_on"), (SPMO_START, "2015-11_on"), ("2018-02-01", "2018-02_on")):
        block = {}
        for name in ("DV2-G", "HPI-G"):
            both = pd.concat([eng[name].loc[start:END], res[name].loc[start:END]], axis=1).dropna()
            block[name] = {"engine": stats(both.iloc[:, 0]), "research": stats(both.iloc[:, 1]),
                           "daily_corr": float(both.corr().iloc[0, 1]),
                           "tracking_error_pp": float((both.iloc[:, 0] - both.iloc[:, 1]).std() * np.sqrt(252) * 100)}
            base = pd.concat([eng_cash[name].loc[start:END], comp[f"{name}|engine|base"].loc[start:END]], axis=1).dropna()
            block[name]["cash_only"] = {"engine": stats(base.iloc[:, 0]), "research": stats(base.iloc[:, 1]),
                                        "daily_corr": float(base.corr().iloc[0, 1])}
            block[name]["parking_effect_cagr_pp"] = {
                "engine": (block[name]["engine"]["cagr"] - block[name]["cash_only"]["engine"]["cagr"]) * 100,
                "research": (block[name]["research"]["cagr"] - block[name]["cash_only"]["research"]["cagr"]) * 100}
            block[name]["research_tbill_parking"] = stats(res[name + "|tbill"].loc[start:END].reindex(both.index))
        cap_e = ev.capsule({"DV2": eng["DV2-G"], "HPI": eng["HPI-G"]}, {"DV2": .5, "HPI": .5}, start=start, end=END)
        cap_r = ev.capsule({"DV2": res["DV2-G"], "HPI": res["HPI-G"]}, {"DV2": .5, "HPI": .5}, start=start, end=END)
        cap_c = ev.capsule({"DV2": eng_cash["DV2-G"], "HPI": eng_cash["HPI-G"]}, {"DV2": .5, "HPI": .5}, start=start, end=END)
        block["capsule"] = {"engine": stats(cap_e), "research": stats(cap_r), "engine_cash_only": stats(cap_c),
                            "daily_corr": float(pd.concat([cap_e, cap_r], axis=1).dropna().corr().iloc[0, 1])}
        excess_ser = (cap_e - rs.cash_rate(cap_e.index)).dropna()
        dsr = deflated_sharpe_ratio(excess_ser.to_numpy(), None, TRIAL_COUNT_FLOAT)
        block["capsule"]["deflated_sharpe_excess_of_tbill"] = {
            "trials": TRIAL_COUNT_FLOAT, "dsr": dsr.deflated_sharpe_float,
            "sharpe_annual": dsr.moments.sharpe_float * np.sqrt(252),
            "benchmark_sharpe_annual": dsr.benchmark_sharpe_float * np.sqrt(252)}
        rep[f"returns_{label}"] = block
    # ---- 3. parking behaviour, costs, engine diagnostics
    keys = ("parking_order_count_dict", "gate_closed_free_slot_session_count_int", "parking_one_way_turnover_per_year_dict",
            "parking_commission_total_dict", "negative_cash_session_int", "min_cash_fraction_float",
            "average_negative_cash_weight_float", "median_cash_fraction_float", "runtime_sec_float", "session_int")
    for pod in ("dv2", "hpi"):
        nav, tx, diag = engine[pod]
        rep[f"{pod}_parking"] = parking_behaviour(nav, tx, vix_gate)
        rep[f"{pod}_diagnostics"] = {"parked": {k: diag.get(k) for k in keys}, "cash": {k: cash[pod][2].get(k) for k in keys}}
    # ---- 6. BIL policy: weekly sweep (B2) vs every-close re-target (earlier DV2 run with the B1 guard)
    p1_folder = OUT / "run2_p1_guard"
    if (p1_folder / "dv2_parked_nav.csv").exists():
        p1_nav, p1_tx, _ = load_engine("dv2", folder=p1_folder)
        p1_ret = p1_nav["total_value"].pct_change().reindex(idx)
        block = {}
        for start in ("2004-01-05", "2007-06-01"):
            both = pd.concat([eng["DV2-G"].loc[start:END], p1_ret.loc[start:END]], axis=1).dropna()
            block[start] = {"weekly_sweep_B2": stats(both.iloc[:, 0]), "every_close_P1": stats(both.iloc[:, 1])}
        block["bil_orders_per_year"] = {"weekly_sweep_B2": rep["dv2_parking"]["orders_per_year"]["BIL"],
                                        "every_close_P1": parking_behaviour(p1_nav, p1_tx, vix_gate)["orders_per_year"]["BIL"]}
        rep["dv2_bil_policy"] = block
    rep["spmo_vs_tbills"] = spmo_vs_tbills(eng, eng_bil, res)
    (OUT / "compare.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
