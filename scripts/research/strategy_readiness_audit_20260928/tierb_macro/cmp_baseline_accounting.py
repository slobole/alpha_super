"""Compass (XLK) and Compass QQQ: baseline reproduction plus A8/A9/A10 accounting checks.

Outputs: OUT/cmp_baseline_accounting.json
- module run_variant (patched only to the frozen audit T5YIE copy) vs the published numbers
- same run through the audit engine helper (must be bit-identical)
- A10 determinism (two runs), A9 capital scaling (30K / 1M / 10M), raw historical share units, 0% withholding
- negative-cash statistics, idle-cash share, fills on padded (zero-volume) bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tb_common as tb

WINDOW_LIST = [
    ("main_2003_05_to_2026_08_19", tb.MAIN_START_STR, tb.MAIN_END_STR),
    ("to_last_bar_2026_09_25", tb.MAIN_START_STR, tb.NORGATE_LAST_BAR_STR),
]


def cash_stats(strategy_obj) -> dict:
    res = strategy_obj.results
    cash = res["cash"].astype(float)
    tv = res["total_value"].astype(float)
    frac = cash / tv
    return {
        "negative_cash_days": int((cash < 0).sum()),
        "days": int(len(cash)),
        "min_cash_frac_nav": float(frac.min()),
        "mean_cash_frac_nav": float(frac.mean()),
        "p01_cash_frac_nav": float(frac.quantile(0.01)),
    }


def padded_fill_count(strategy_obj, execution_price_df: pd.DataFrame) -> dict:
    tx = strategy_obj.get_transactions()
    tx = tx[tx["order_id"] != -1] if "order_id" in tx.columns else tx
    n_pad = 0
    rows = []
    for _, row in tx.iterrows():
        bar = pd.Timestamp(row["bar"])
        asset = str(row["asset"])
        vol = float(execution_price_df.loc[bar, (asset, "Volume")]) if (asset, "Volume") in execution_price_df.columns else np.nan
        if not np.isfinite(vol) or vol <= 0:
            n_pad += 1
            rows.append({"bar": str(bar.date()), "asset": asset, "volume": vol})
    return {"fills": int(len(tx)), "fills_on_zero_or_nan_volume": n_pad, "examples": rows[:10]}


def main() -> None:
    out: dict = {}
    for variant_str, module_obj in (("xlk", tb.cmp_mod), ("qqq", tb.cmpq_mod)):
        tb.compass_config(variant_str)
        vout: dict = {}
        # module path (run_variant) on the main window
        strat_mod = module_obj.run_variant(
            show_display_bool=False, save_results_bool=False,
            backtest_start_date_str=tb.MAIN_START_STR, end_date_str=tb.MAIN_END_STR,
            config_obj=module_obj.DEFAULT_CONFIG,
        )
        nav_mod = tb.nav_ser(strat_mod)
        vout["module_run_variant_main"] = tb.metrics(nav_mod)
        # audit helper on the same inputs must be bit-identical
        strat_a = tb.run_compass_engine(variant_str, end_date_str=tb.MAIN_END_STR)
        nav_a = tb.nav_ser(strat_a)
        vout["helper_equals_module_max_abs_nav_diff"] = float((nav_a - nav_mod).abs().max())
        # determinism
        strat_b = tb.run_compass_engine(variant_str, end_date_str=tb.MAIN_END_STR)
        vout["determinism_bit_identical"] = bool(tb.nav_ser(strat_b).equals(nav_a))
        for win_str, start_str, end_str in WINDOW_LIST:
            s = tb.run_compass_engine(variant_str, end_date_str=end_str)
            nav = tb.nav_ser(s)
            wout = {"base_100k": tb.metrics(nav, start_str, end_str), "cash": cash_stats(s)}
            wout["last3y_2023_09_25"] = tb.metrics(nav, "2023-09-25", end_str) if end_str > "2023-10-01" else None
            exec_df = tb.compass_execution_price_df(variant_str, end_str)
            wout["padded_fills"] = padded_fill_count(s, exec_df)
            wout["orders_per_year"] = float(
                len(s.get_transactions()) / ((nav.index[-1] - nav.index[0]).days / 365.25)
            )
            if win_str != "main_2003_05_to_2026_08_19":
                vout[win_str] = wout
                continue
            # capital scaling
            cap_dict = {}
            for cap in (30_000.0, 1_000_000.0, 10_000_000.0):
                sc = tb.run_compass_engine(variant_str, end_date_str=end_str, capital_base_float=cap)
                cap_dict[str(int(cap))] = tb.metrics(tb.nav_ser(sc), start_str, end_str)
                cap_dict[str(int(cap)) + "_daily_ret_corr_vs_100k"] = float(
                    np.corrcoef(tb.nav_ser(sc).pct_change().dropna(), nav.pct_change().dropna())[0, 1]
                )
            wout["capital_scaling"] = cap_dict
            # raw historical share units (commission is 0, so only rounding moves)
            sr = tb.run_compass_engine(variant_str, end_date_str=end_str, historical_share_units_bool=True)
            wout["historical_share_units"] = tb.metrics(tb.nav_ser(sr), start_str, end_str)
            # withholding sensitivity (house default 25%)
            s0 = tb.run_compass_engine(variant_str, end_date_str=end_str, withholding_rate_float=0.0)
            wout["withholding_0pct"] = tb.metrics(tb.nav_ser(s0), start_str, end_str)
            wout["dividend_totals_base"] = {
                "gross": float(s.dividend_cash_gross_total_float),
                "withheld": float(s.dividend_withholding_total_float),
                "withholding_rate": float(s.dividend_withholding_rate_float),
            }
            vout[win_str] = wout
        out[variant_str] = vout
        print(variant_str, vout["module_run_variant_main"], flush=True)
    tb.write_json("cmp_baseline_accounting.json", out)


if __name__ == "__main__":
    main()
