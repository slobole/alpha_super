"""Phase 4: frozen DV2 rules on other universes (S&P 400, S&P 600) and on the pre-declared equity-ETF list."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp  # noqa: E402
import batch  # noqa: E402

OUT = batch.OUT
GROUPS = {
    "sectors": "XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE XLC".split(),
    "industries": "XBI IBB SMH SOXX KRE KBE XHB ITB XRT XOP OIH XME GDX IYT IGV ITA IHI XSD XPH".split(),
    "countries": "EWJ EWG EWU EWC EWA EWZ EWH EWT EWY EWW EWS EWP EWQ EWI EWL EWN FXI INDA EZA EEM EFA".split(),
    "broad": "SPY QQQ IWM DIA MDY IWD IWF IWN IWO".split(),
}


def etf_panel(subset=None) -> rp.Panel:
    p = rp.Panel("etf")
    hist = np.cumsum(np.isfinite(p.C), axis=0)
    member = hist >= 252  # own history >= 252 sessions (warm-up)
    if subset is not None:
        keep = np.array([s in subset for s in p.symbols])
        member = member & keep[None, :]
    p.member = member
    return p


def run_block(p, rules, start="2000-01-03", end="2026-08-19"):
    out = {}
    for name, r in rules.items():
        res = rp.run(p, r, start=start, end=end)
        out[name] = (rp.summarize(res), rp.daily_returns(res))
    return out


def main():
    report = {}
    sp = rp.Panel("sp500")
    base_ret = rp.daily_returns(rp.run(sp, rp.Rule(floor=True)))
    spx_ret = pd.Series(sp.spx, index=sp.dates).pct_change()
    del sp
    wired, cost10 = rp.Rule(), rp.Rule(slippage=0.0005)
    # ETFs
    for gname, subset in [("all_etf", None)] + list(GROUPS.items()):
        p = etf_panel(subset)
        res = run_block(p, {"engine_costs": wired, "10bps_rt": cost10, "slots5_engine": wired.with_(slots=5)})
        report[f"etf_{gname}"] = {}
        for k, (s, r) in res.items():
            both = pd.concat([r, base_ret, spx_ret.reindex(r.index)], axis=1).dropna()
            s["corr_with_stock_floor"] = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
            s["corr_with_spx"] = float(both.iloc[:, 0].corr(both.iloc[:, 2]))
            b = np.cov(both.iloc[:, 0], both.iloc[:, 2])[0, 1] / both.iloc[:, 2].var()
            resid = both.iloc[:, 0] - b * both.iloc[:, 2]
            s["beta_spx"] = float(b)
            s["alpha_ann_vs_spx"] = float(resid.mean() * 252)
            report[f"etf_{gname}"][k] = s
            if gname == "all_etf" and k == "engine_costs":
                r.to_frame("etf_dv2").to_parquet(OUT / "phase4_etf_all_returns.parquet")
        print(gname, {k: round(v["sharpe"], 3) for k, v in report[f"etf_{gname}"].items()}, flush=True)
    # other stock universes (exact rules; floor uses that universe's own member median)
    for label in ("sp400", "sp600"):
        p = rp.Panel(label)
        res = run_block(p, {"wired": wired, "floor": rp.Rule(floor=True), "wired_10bps_rt": cost10, "rank_adv_floor": rp.Rule(floor=True, rank="adv")})
        report[label] = {}
        for k, (s, r) in res.items():
            both = pd.concat([r, base_ret], axis=1).dropna()
            s["corr_with_sp500_floor"] = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
            report[label][k] = s
        print(label, {k: round(v["sharpe"], 3) for k, v in report[label].items()}, flush=True)
        del p
    (OUT / "phase4_universes.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
