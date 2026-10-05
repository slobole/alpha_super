"""Fund products, final pass: exposure look-through and the one-day gap table (SPEC 3, "Exposure look-through").

Per product (at target capital weights, per unit of product NAV):
    Nasdaq-100 look-through notional  N_t = w_TAA x 3 x TQQQ weight_t + w_MOM x momentum invested weight_t
    equity exposure for the gap table  E_t = N_t + w_MR x MR stock weight_t          (MR stocks at beta 1)
    T-bill-like share                  B_t = BIL sleeve + w_MR x (1 - MR stock weight_t) + w_MOM x (1 - invested_t)
                                             + w_TAA x TAA idle cash_t

TQQQ weight inside a TAA pod = TQQQ shares held (cumulative fills) x CAPITALSPECIAL close / pod NAV, daily, on the
real-data era only (2012-10-02 -> END): the 2008-2012 proxy era has synthetic TQQQ bars that are not stored as a
price series, so it is left out (amendment I1 of the plan).
The gap table is arithmetic, not a simulation: product loss = E x fall for the TAA / MOM / MR terms as defined
above (TQQQ = 3 x the index move, MOM beta 1, MR stocks beta 1, every other asset flat).

Usage: PYTHONDONTWRITEBYTECODE=1 python exposure.py   (after study.py). Writes <study>/report/exposure.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import g_lib as g
from g_lib import END, EXACT_START, TBILL, Lab, lib

FALLS = (0.05, 0.10, 0.15, 0.20)


def tqqq_weight(lab: Lab, alias: str, close: pd.Series) -> pd.Series:
    tx = lab.data["tx"][alias]
    t = tx[tx["asset_str"] == "TQQQ"].groupby("date")["amount_float"].sum()
    nav = lab.data["nav"][alias]
    shares = t.reindex(nav.index).fillna(0.0).cumsum()          # shares held after each session's fills
    w = (shares * close.reindex(nav.index)) / nav
    # sanity: fill notional equals shares x fill price, and the weight stays inside [0, 1.05]
    chk = tx[tx["asset_str"] == "TQQQ"]
    assert np.allclose(chk["amount_float"] * chk["fill_price_float"], chk["signed_notional_float"], rtol=1e-6)
    assert w.loc[EXACT_START:END].between(-1e-9, 1.05).all(), (alias, float(w.loc[EXACT_START:END].max()))
    return w.loc[EXACT_START:END]


def dist(s: pd.Series) -> dict:
    return {"mean": float(s.mean()), "p90": float(s.quantile(0.90)), "max": float(s.max()), "min": float(s.min())}


def main() -> int:
    import norgatedata  # noqa: PLC0415

    lab = Lab()
    data = lab.data
    study = json.loads((g.OUT / "study.json").read_text(encoding="utf-8"))
    px = norgatedata.price_timeseries("TQQQ", stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
                                      padding_setting=norgatedata.PaddingType.NONE, start_date="2010-01-01",
                                      end_date=END.strftime("%Y-%m-%d"), timeseriesformat="pandas-dataframe")
    close = px["Close"]
    close.index = pd.to_datetime(close.index).normalize()
    idx = data["nav"]["taa3x"].loc[EXACT_START:END].index

    tq = {a: tqqq_weight(lab, a, close) for a in ("taa3x", "taa3x_1n")}
    inv = lambda path: (path["portfolio_value_float"] / path["total_value_float"]).reindex(idx)  # noqa: E731
    mom_inv = 0.5 * inv(data["path"]["ndx_atr_cap"]) + 0.5 * inv(data["path"]["ndx_natr_cap"])
    mr_stock = 0.5 * inv(data["path"]["dv2_g_cash"]) + 0.5 * inv(data["path"]["hpi_g_cash"])   # parking-off runs: stocks only
    taa_cash = {a: (lib.read_path(lib.SOURCE, a)["cash_float"] / lib.read_path(lib.SOURCE, a)["total_value_float"]).reindex(idx) for a in tq}
    vix = norgatedata.price_timeseries("$VIX", start_date="1990-01-01", end_date=END.strftime("%Y-%m-%d"), timeseriesformat="pandas-dataframe")["Close"]
    vix.index = pd.to_datetime(vix.index).normalize()
    gate_prev = g.vix_gate.stress_gate_open_ser(vix).shift(1).reindex(idx)    # *** CRITICAL*** state after close t-1 governs session t

    month_end = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().to_numpy()
    out: dict = {"meta": {"window": [str(idx[0].date()), str(idx[-1].date())], "note": "real-data era only; the proxy era has no stored TQQQ price series"},
                 "tqqq_weight": {a: {"daily": dist(s), "month_end": dist(s.loc[month_end]), "share_of_days_above_50pct": float((s > 0.5).mean()),
                                     "share_of_days_zero": float((s < 0.01).mean())} for a, s in tq.items()},
                 "mom_invested": dist(mom_inv), "mr_stock_weight": {**dist(mr_stock), "gate_open": float(mr_stock[gate_prev == True].mean()),  # noqa: E712
                                                                    "gate_closed": float(mr_stock[gate_prev == False].mean())},  # noqa: E712
                 "gate_open_share": float((gate_prev == True).mean()), "books": {}}  # noqa: E712

    books: dict[str, tuple[dict, float]] = {}
    for n in g.PRODUCTS:
        books[n] = (g.PRODUCTS[n], 1.0)
        final = study["products"][n]["final"]
        if final != n:
            books[final] = (g.with_cash(g.PRODUCTS[n], study["products"][n]["cash_added"]), 1.0)
    for name in study["dial"]:
        books[name] = (g.blend((1.0, study["books"][name]["weights"])), 1.0)
    for key, row in study["margin"].items():
        for kind in ("vol_matched", "cagr_matched", "fixed_140"):
            if row.get(kind):
                books[row[kind]["name"]] = (g.PRODUCTS[key.split(" -> ")[0]], float(row[kind]["L"]))
    for name, (w, L) in books.items():
        w_taa = {a: w.get(a, 0.0) for a in ("taa3x", "taa3x_1n")}
        w_mom = w.get("ndx_atr_cap", 0.0) + w.get("ndx_natr_cap", 0.0)
        w_mr = w.get("dv2_g", 0.0) + w.get("hpi_g", 0.0)
        nasdaq = L * (sum(w_taa[a] * 3.0 * tq[a] for a in tq) + w_mom * mom_inv)
        # the same with the prior-close weights the book carried between annual resets (pods drift inside a year)
        wp: list = []
        g.book_returns(lab.frame, g.blend((1.0, w)), lab.start, weight_path=wp)
        widx, wcols, wmat = wp[0]
        wdf = pd.DataFrame(wmat, index=widx, columns=wcols).reindex(idx)
        col = lambda a: wdf[a] if a in wdf else 0.0  # noqa: E731
        inv_pod = {a: inv(data["path"][a]) for a in ("ndx_atr_cap", "ndx_natr_cap")}
        nasdaq_drift = L * (sum(col(a) * 3.0 * tq[a] for a in tq) + sum(col(a) * inv_pod[a] for a in inv_pod))
        equity = nasdaq + L * w_mr * mr_stock
        tbill_like = w.get(TBILL, 0.0) + w_mr * (1.0 - mr_stock) + w_mom * (1.0 - mom_inv) + sum(w_taa[a] * taa_cash[a].clip(lower=0.0) for a in tq)
        e = dist(equity)
        out["books"][name] = {
            "L": L, "nasdaq_lookthrough_drift": {**dist(nasdaq_drift), "peak_date": str(nasdaq_drift.idxmax().date()),
                                                 "max_since_2015": float(nasdaq_drift.loc["2015-01-01":].max())},
            "nasdaq_peak_date": str(nasdaq.idxmax().date()), "nasdaq_max_since_2015": float(nasdaq.loc["2015-01-01":].max()),
            "nasdaq_lookthrough": {**dist(nasdaq), "by_year_max": {int(y): float(v) for y, v in nasdaq.groupby(nasdaq.index.year).max().items()},
                                           "by_year_mean": {int(y): float(v) for y, v in nasdaq.groupby(nasdaq.index.year).mean().items()}},
            "equity_exposure": e,
            "gap_table": {f"{int(f * 100)}%": {"mean": -f * e["mean"], "p90": -f * e["p90"], "peak": -f * e["max"]} for f in FALLS},
            "tbill_like_share": {**dist(tbill_like), "gate_open": float(tbill_like[gate_prev == True].mean()),  # noqa: E712
                                 "gate_closed": float(tbill_like[gate_prev == False].mean())},  # noqa: E712
            "nasdaq_monthly": [[d.strftime("%Y-%m"), round(float(v), 3)] for d, v in nasdaq.resample("ME").last().items()]}
        print(name, "Nasdaq look-through mean/p90/max", [round(x, 2) for x in (dist(nasdaq)["mean"], dist(nasdaq)["p90"], dist(nasdaq)["max"])],
              "gap 10% at peak", round(-0.10 * e["max"], 3), flush=True)
    (g.OUT / "exposure.json").write_text(json.dumps(g.r6(out), indent=1, default=str), encoding="utf-8")
    g.ledger("exposure_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
