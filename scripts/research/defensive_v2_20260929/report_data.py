"""Collect the defensive v2 results (first run, A1, A2 checks, friction) into one JSON for the report page.

Usage: python report_data.py  ->  results/.../defensive_v2_20260929/report/report_data.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "results" / "research" / "portfolio" / "defensive_v2_20260929"
CHK = OUT / "sharpe_checks"
PICK, TOP, NOEOM, CHAMP, DPRIME, BENCH = ("CORE5 + EOM + DISP [IV]", "CORE5 + EOM + DV2-IND [EQ] + G3 10%",
                                          "BTAL_QQQ + DV2-IND [EQ]", "CORE5 60 + BTAL_QQQ 40",
                                          "CORE5 + BTAL_QQQ [IV]", "CORE5 [EQ]")
REC = "CORE5 + BTAL_QQQ + EOM [IV]"  # A3: SPEC-literal tie-break pick
DV2B = "CORE5 + BTAL_QQQ + DV2-IND [IV]"  # owner candidate (EOM looks weak lately)
BOTH = "CORE5 + BTAL_QQQ + EOM + DV2-IND [EQ]"  # both edges, a quarter each
CURVES = [BOTH, REC, DV2B, CHAMP]


def main() -> int:
    checks = json.loads((CHK / "checks.json").read_text(encoding="utf-8"))
    R = pd.read_csv(OUT / "main_long_returns.csv.gz", index_col=0, parse_dates=True)
    books = pd.read_csv(OUT / "main_books_sharpe.csv", index_col=0)
    house = pd.read_csv(CHK / "house_main_books.csv", index_col=0)
    periods = pd.read_csv(CHK / "periods.csv")
    years = pd.read_csv(CHK / "calendar_excess.csv", index_col=0)
    crises = pd.read_csv(CHK / "crises.csv", index_col=0)
    plateau = pd.read_csv(CHK / "plateau.csv")
    cap = pd.read_csv(CHK / "capacity.csv", index_col=0)
    fric = pd.read_csv(OUT / "friction" / "friction_by_product.csv")

    nav = (1 + R[CURVES]).cumprod()
    dd = nav / nav.cummax() - 1
    m_nav = nav.resample("ME").last()
    m_dd = dd.resample("ME").min()
    curves = {"dates": [d.strftime("%Y-%m") for d in m_nav.index],
              "nav": {b: [round(float(v), 4) for v in m_nav[b]] for b in CURVES},
              "dd": {b: [round(float(v), 4) for v in m_dd[b]] for b in CURVES}}

    table_books = [REC, PICK, TOP, NOEOM, CHAMP, DPRIME, BENCH]
    pv = periods.set_index(["book", "cash", "window"])
    rows = []
    for b in table_books:
        get = lambda cash, w, k: float(pv.loc[(b, cash, w), k])  # noqa: E731
        rows.append({"book": b, "long_cagr": get("fair", "LONG", "cagr"), "long_sharpe": get("fair", "LONG", "sharpe"),
                     "long_xs_sharpe": get("fair", "LONG", "xs_sharpe"), "long_maxdd": get("fair", "LONG", "maxdd"),
                     "exact_cagr": get("fair", "EXACT", "cagr"), "exact_maxdd": get("fair", "EXACT", "maxdd"),
                     "recent_cagr": get("fair", "RECENT", "cagr"), "recent_sharpe": get("fair", "RECENT", "sharpe"),
                     "house_sharpe": get("house", "LONG", "sharpe"), "house_cagr": get("house", "LONG", "cagr"),
                     "p_breach": float(books.at[b, "p_breach10"]), "house_p_breach": float(house.at[b, "p_breach10"]),
                     "gates": bool(books.at[b, "gates_pass"]), "house_gates": bool(house.at[b, "gates_pass"]),
                     "trade_days": float(books.at[b, "trade_days_per_year"]),
                     "weights": json.loads(books.at[b, "avg_weights"])})
    sleeves = []
    for s in ["core5", "btal_qqq", "eom_flow", "disp", "etf_dv2", "tbill"]:
        name = f"sleeve:{s}"
        sleeves.append({"sleeve": s, **{f"{w}_{k}": float(pv.loc[(name, "fair", w), k])
                                         for w in ["A", "B", "C", "RECENT", "LONG"] for k in ["cagr", "sharpe", "maxdd"]},
                        "house_RECENT_cagr": float(pv.loc[(name, "house", "RECENT"), "cagr"])})

    yr = {k: {str(int(y)): round(float(v), 4) for y, v in years[f"fair|{k}"].items()}
          for k in [REC, PICK, CHAMP, NOEOM, "sleeve:eom_flow"]}
    crisis_keys = [c for c in crises.columns]
    cr = {b: {k: float(crises.at[b, k]) for k in crisis_keys} for b in [REC, PICK, CHAMP, NOEOM, BENCH, "sleeve:eom_flow"]}

    pl = plateau.to_dict(orient="records")
    capacity = {b: {k: (None if pd.isna(v) else v) for k, v in cap.loc[b].items()} for b in [REC, PICK, NOEOM, CHAMP, TOP]
                if b in cap.index}
    friction = fric.to_dict(orient="records")
    followups = json.loads((CHK / "review_followups.json").read_text(encoding="utf-8"))
    breakeven = pd.read_csv(CHK / "eom_cost_breakeven.csv").to_dict(orient="records")
    turnover = json.loads((CHK / "eom_turnover.json").read_text(encoding="utf-8"))["eom_turnover_per_year"]
    plateau_rec = pd.read_csv(CHK / "plateau_runner_up.csv").to_dict(orient="records")
    dossier = json.loads((CHK / "candidate_dossier.json").read_text(encoding="utf-8"))
    cap_c = pd.read_csv(CHK / "capacity_candidates.csv", index_col=0)
    capacity_c = {b: {k: (None if pd.isna(v) else v) for k, v in cap_c.loc[b].items()} for b in cap_c.index}
    plateau_c = pd.read_csv(CHK / "plateau_candidates.csv").to_dict(orient="records")
    data = {"checks": checks, "curves": curves, "table": rows, "sleeves": sleeves, "years": yr, "crises": cr,
            "plateau": pl, "capacity": capacity, "friction": friction, "followups": followups,
            "breakeven": breakeven, "eom_turnover": turnover, "plateau_rec": plateau_rec,
            "dossier": dossier, "capacity_c": capacity_c, "plateau_c": plateau_c,
            "gates": {b: {"gates": bool(books.at[b, "gates_pass"]), "p_breach": float(books.at[b, "p_breach10"]),
                          "r": {g: bool(books.at[b, g]) for g in ["r1", "r2", "r3", "r4", "r5"]},
                          "slot_fail": (None if pd.isna(books.at[b, "slot_fail"]) else str(books.at[b, "slot_fail"])),
                          "house_gates": bool(house.at[b, "gates_pass"]),
                          "house_p_breach": (None if pd.isna(house.at[b, "p_breach10"]) else float(house.at[b, "p_breach10"])),
                          "trade_days": float(books.at[b, "trade_days_per_year"]),
                          "weights": json.loads(books.at[b, "avg_weights"])}
                      for b in [BOTH, DV2B, REC, PICK, CHAMP, BENCH, TOP, NOEOM, DPRIME]},
            "names": {"both": BOTH, "dv2b": DV2B, "rec": REC, "pick": PICK, "top": TOP, "noeom": NOEOM, "champ": CHAMP, "dprime": DPRIME, "bench": BENCH}}
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    data = clean(data)
    (OUT / "report").mkdir(exist_ok=True)
    (OUT / "report" / "report_data.json").write_text(json.dumps(data, default=float, allow_nan=False), encoding="utf-8")
    print(len(json.dumps(data, default=float)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
