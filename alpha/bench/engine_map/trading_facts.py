"""How each engine trades, for the trading row in the Engine Map's drawer. Writes results/bench/engine_map/_work/trading_facts.json.
Run it with python -m alpha.bench.engine_map trading-facts.

Reads each engine's newest saved research run with the strategy classes stubbed: no norgatedata and no backtest. Same
measures as the panel's trading_spec.py, on the last 3 years of the run, with T-bill parking left out. It runs by hand,
not nightly, because the values barely move. A missing run or value stays null; build.py and the page read null as
not measured, so nothing here can stop a build.

    days        trade dates a year
    turnover    buys plus sells a year, over NAV
    names       median number of positions held at a close; namesMax is the most at any close
    hold        median stretch on one side of one asset, in trading sessions (rows of the daily weight frame), for stretches that began in the 3 years;
                a flip from long to short starts a new one
    shortPct    size of the short book on days it is short, median, percent of capital; 0 means no shorts
    shortDays   percent of sessions holding a short
    end         last date of the saved run

A run with no weight frame (Trinity) answers names, hold and shorts from its cumulative fills. The MR capsule has no entry: it is half DV2-G and half
HPI-G, each with its own.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from alpha.bench.engine_map.refresh import WORK, saved_run

OUT = WORK / "trading_facts.json"
CASHLIKE = {"Cash", "BIL", "SHV", "SGOV"}
ENGINE_STEMS = {
    "taa3x": "strategy_taa_df_btal_fallback_tqqq_vix_cash", "taa3x_1n": "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "taa2x": "strategy_taa_df_btal_1n_fallback_qld_vix_cash",
    "btal_qqq": "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "ndx_vxn": "strategy_mo_atr_normalized_ndx_vxn_scaled", "ndx_atr": "strategy_mo_atr_normalized_ndx",
    "ndx_cap_atr": "strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap", "ndx_cap_natr": "strategy_mo_natr20_ndx_vxn_scaled_sector_cap", "dv2": "strategy_mr_dv2", "dv2_g": "strategy_mr_dv2_vix_gated_bil",
    "hpi": "strategy_mr_hpi_sp500_2_3_5_vote", "hpi_g": "strategy_mr_hpi_vote_vix_gated_bil", "hpi_rsi": "strategy_mr_hpi_sp500_ibs_rsi_exit", "dv2_ind": "strategy_mr_dv2_industry_etf",
    "downshock": "strategy_mr_us_sector_etf_ibs_downshock_vox_iyr", "dispersion": "strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200", "core5": "strategy_taa_adaptive_macro_core5",
    "trinity": "strategy_taa_trinity_vol_control_8_bil", "compass": "strategy_taa_inflation_compass", "tfi": "strategy_taa_tactical_fixed_income_ief_lqd", "eom": "strategy_taa_month_end_rebalancing_flow",
    "eom_vote": "strategy_taa_month_end_flow_vote", "dv2_g_liq": "strategy_mr_dv2_vix_gated_liq_bil", "hpi_g_liq": "strategy_mr_hpi_vote_vix_gated_liq_bil",
    "mo_leg1": "strategy_mo_capsule_leg1_natr_ndx", "mo_leg2": "strategy_mo_capsule_leg2_residual_ndx", "mo_leg3": "strategy_mo_capsule_leg3_residual_r1000", "vrp": "strategy_vrp_svxy_vixm_rsi2",
    "t1": "strategy_taa_beyond_6040_trend_tilt",
}
YEARS = 3.0


def number(value) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


def stretches(weights: pd.DataFrame, since: pd.Timestamp) -> list[int]:
    """Length in rows of every unbroken stretch on one side of one asset that began on or after `since`. A flip from long to short starts a new one."""
    lengths: list[int] = []
    for column in weights.columns:
        side = np.sign(weights[column].where(weights[column].abs() > 1e-4, 0.0)).to_numpy().astype(int)
        changes = np.flatnonzero(np.diff(np.r_[0, side, 0]))
        for begin, stop in zip(changes[:-1], changes[1:]):
            if side[begin] != 0 and weights.index[begin] >= since:
                lengths.append(int(stop - begin))
    return lengths


def shares_from_fills(fills: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Shares held at each close, for a run that saved no weight frame. Sizes mean nothing here, only which assets and on which side."""
    return fills.pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum").reindex(index).fillna(0.0).cumsum()


def measure(stem: str) -> dict:
    run, state = saved_run(stem)
    nav = state["results"]["total_value"]
    start, end = nav.index[0], nav.index[-1]
    since = end - pd.DateOffset(years=3)
    fills = state["_transactions"].copy()
    fills["bar"] = pd.to_datetime(fills["bar"])
    fills["share"] = fills["total_value"].abs() / nav.reindex(fills["bar"]).to_numpy()
    traded = fills[(fills["bar"] >= since) & ~fills["asset"].isin(CASHLIKE)]
    row = {
        "run": run[:10], "start": str(start.date()), "end": str(end.date()),
        "days": number(traded["bar"].nunique() / YEARS), "turnover": number(traded["share"].sum() / YEARS),
        "names": None, "namesMax": None, "hold": None, "shortPct": None, "shortDays": None,
    }
    weights = state.get("realized_weight_df")
    from_fills = weights is None or not len(weights)
    if from_fills:
        weights = shares_from_fills(fills, nav.index)
    weights = weights.fillna(0.0).drop(columns=[c for c in weights.columns if c in CASHLIKE], errors="ignore")
    recent = weights.loc[since:]
    held = (recent.abs() > 1e-4).sum(axis=1)
    shorts = recent.where(recent < -1e-4)
    short_book = shorts.abs().sum(axis=1)
    lengths = stretches(weights, since)
    has_shorts = bool((short_book > 0).any())
    row.update({
        "names": number(held.median()), "namesMax": number(held.max()), "hold": number(float(np.median(lengths))) if lengths else None,
        # share counts give no percent of capital, so a run read from fills can only say none
        "shortPct": 0.0 if not has_shorts else None if from_fills else number(short_book[short_book > 0].median() * 100),
        "shortDays": number((recent < -1e-4).any(axis=1).mean() * 100),
    })
    return row


def main() -> int:
    facts = {}
    for engine_id, stem in ENGINE_STEMS.items():
        try:
            facts[engine_id] = measure(stem)
        except Exception as exc:  # one unreadable run leaves that engine not measured, and says so
            print("not measured", engine_id, repr(exc)[:200], flush=True)
    assert "norgatedata" not in sys.modules
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(pd.DataFrame(facts).T.drop(columns=["start"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
