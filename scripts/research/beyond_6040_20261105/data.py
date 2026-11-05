"""Panel builder for the Beyond 60/40 research (2026-11-05). Research-only.

Reuses the audited PMT panel (stocks / 25y bond / gold / Fed funds, 1968→) and adds, with the same
on-or-before alignment rule (never backward fill):
  r_bond10   synthetic 10y par Treasury off DGS10 → IEF TR from 2002-07-30
  r_world    French Developed ex-US market (Mkt-RF + RF, USD) 1990-07 → EFA TR from 2001-08-20
  r_dbc, r_tip, r_vt, r_bil   ETF total returns (NaN before listing)
  tbill      DTB3 / 100 (annual)
  cpi_yoy    CPI year-over-year, lagged one month (released mid-next-month)
  usdils     Norgate USDILS spot close (1991-01 →)
  il_rate    OECD immediate rate Israel (monthly, annual %), lagged one month
Writes out/panel.parquet and out/panel_meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RAW = OUT / "raw"
PMT_PANEL = Path(r"C:\Users\User\Documents\workspace\pakal\pakal-research\reports"
                 r"\beyond_passive_poor_mans_trend_study\data\panel.parquet")
IEF_START = pd.Timestamp("2002-07-30")
EFA_START = pd.Timestamp("2001-08-20")


def read_fred(name: str) -> pd.Series:
    df = pd.read_csv(RAW / f"{name}.csv")
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def on_or_before(s: pd.Series, sessions: pd.DatetimeIndex, limit: int = 7) -> pd.Series:
    # *** CRITICAL *** carry the last known value forward only (never backward)
    return s.sort_index().reindex(s.index.union(sessions)).ffill(limit=limit).reindex(sessions)


def par_bond_return(y_prev, y_now, dt_years, maturity):
    n = maturity - dt_years
    h = y_now / 2.0
    disc = (1.0 + h) ** (-2.0 * n)
    price = np.where(np.abs(y_now) > 1e-9, y_prev / y_now * (1.0 - disc) + disc, 1.0 + y_prev * n)
    return price - 1.0 + y_prev * dt_years


def synthetic_10y(sessions: pd.DatetimeIndex, maturity: float = 10.0) -> pd.Series:
    y = on_or_before(read_fred("DGS10") / 100.0, sessions, limit=3)
    dt = np.r_[np.nan, np.diff(sessions.values).astype("timedelta64[D]").astype(float)] / 365.25
    r = par_bond_return(y.shift(1).values, y.values, dt, maturity)
    return pd.Series(r, index=sessions, name="bond10_syn")


def french_dev_ex_us() -> pd.Series:
    rows = []
    with open(RAW / "Developed_ex_US_3_Factors_Daily.csv", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 8:
                rows.append((parts[0], float(parts[1]), float(parts[4])))
    df = pd.DataFrame(rows, columns=["d", "mkt_rf", "rf"])
    df["d"] = pd.to_datetime(df["d"], format="%Y%m%d")
    df = df.set_index("d")
    df = df[(df["mkt_rf"] > -90) & (df["rf"] > -90)]
    return (df["mkt_rf"] + df["rf"]) / 100.0


def fetch_norgate(start: str, syms: list[str]) -> dict[str, pd.Series]:
    import norgatedata as nd
    out = {}
    for sym in syms:
        kw = dict(padding_setting=nd.PaddingType.NONE, start_date=start, timeseriesformat="pandas-dataframe")
        if not sym.startswith("$") and sym not in ("USDILS",):
            kw["stock_price_adjustment_setting"] = nd.StockPriceAdjustmentType.TOTALRETURN
        out[sym] = nd.price_timeseries(sym, **kw)["Close"]
    return out


def build() -> pd.DataFrame:
    base = pd.read_parquet(PMT_PANEL)
    sessions = pd.DatetimeIndex(base.index)
    ret = lambda s: s.reindex(sessions).pct_change(fill_method=None)  # noqa: E731

    ng = fetch_norgate("1968-01-01", ["IEF", "EFA", "DBC", "TIP", "VT", "BIL", "USDILS"])

    # --- 10y bond
    syn10 = synthetic_10y(sessions)
    ief = ret(ng["IEF"])
    bond10 = syn10.where(sessions < IEF_START, ief)

    # --- world ex-US stocks (USD)
    fr = french_dev_ex_us()
    fr_s = fr.reindex(sessions)  # French daily is on US calendar; missing days -> NaN (treated as 0 below, counted)
    efa = ret(ng["EFA"])
    world = fr_s.where(sessions < EFA_START, efa)

    tbill = on_or_before(read_fred("DTB3") / 100.0, sessions, limit=10)
    cpi = read_fred("CPIAUCSL")
    cpi_yoy = (cpi / cpi.shift(12) - 1.0)
    # availability: CPI for month m is published mid m+1; make it known from the first session of m+2 (conservative)
    cpi_yoy.index = cpi_yoy.index + pd.offsets.MonthBegin(2)
    cpi_yoy_s = on_or_before(cpi_yoy, sessions, limit=40)
    il = read_fred("IRSTCI01ILM156N") / 100.0
    il.index = il.index + pd.offsets.MonthBegin(2)
    il_s = on_or_before(il, sessions, limit=40)
    usdils = on_or_before(ng["USDILS"], sessions, limit=5)

    panel = base[["r_stock", "r_bond", "r_gold", "dff", "src_etf", "vix"]].copy()
    panel["r_bond10"] = bond10
    panel["r_world"] = world
    panel["r_dbc"] = ret(ng["DBC"])
    panel["r_tip"] = ret(ng["TIP"])
    panel["r_vt"] = ret(ng["VT"])
    panel["r_bil"] = ret(ng["BIL"])
    panel["tbill"] = tbill.ffill()
    panel["cpi_yoy"] = cpi_yoy_s
    panel["il_rate"] = il_s
    panel["usdils"] = usdils

    meta = {"rows": len(panel), "start": str(panel.index[0].date()), "end": str(panel.index[-1].date())}
    nan_counts = {}
    for c in ["r_bond10", "r_world"]:
        first = panel[c].first_valid_index()
        n = int(panel.loc[first:, c].isna().sum())
        nan_counts[c] = {"first_valid": str(first.date()), "nan_after_first_filled_zero": n}
        panel.loc[first:, c] = panel.loc[first:, c].fillna(0.0)
    meta["nan_fill"] = nan_counts
    # splice checks
    j = pd.concat([syn10, ief], axis=1, keys=["syn", "ief"]).dropna().loc["2002-08-01":]
    meta["bond10_vs_ief"] = {"corr": float(j.corr().iloc[0, 1]), "vol_syn": float(j["syn"].std() * 252 ** 0.5),
                             "vol_ief": float(j["ief"].std() * 252 ** 0.5), "ann_ret_syn": float(j["syn"].mean() * 252),
                             "ann_ret_ief": float(j["ief"].mean() * 252)}
    j = pd.concat([fr_s, efa], axis=1, keys=["fr", "efa"]).dropna().loc["2001-09-01":]
    meta["world_french_vs_efa"] = {"corr": float(j.corr().iloc[0, 1]), "vol_fr": float(j["fr"].std() * 252 ** 0.5),
                                   "vol_efa": float(j["efa"].std() * 252 ** 0.5), "n": int(len(j))}
    try:
        import norgatedata as nd
        meta["norgate_update"] = {db: str(nd.last_database_update_time(db)) for db in ["US Equities", "Forex Spot"]}
    except Exception:
        pass
    panel.index.name = "date"
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "panel.parquet")
    (OUT / "panel_meta.json").write_text(json.dumps(meta, indent=2))
    return panel


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT / "panel.parquet")


if __name__ == "__main__":
    p = build()
    print(p.shape, p.index[0].date(), p.index[-1].date())
    print(json.dumps(json.loads((OUT / "panel_meta.json").read_text()), indent=1))
