"""Analyze the young-client runs and apply the pre-declared decision rules (plan_frozen.yaml)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH.parents[0] / "fund_menu_20260923"))
import common  # noqa: E402  (fund-menu metric helpers)

STUDY_DIR_PATH = common.REPO_ROOT_PATH / "results" / "research" / "portfolio" / "young_client_taa_core5_20260923"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
WIRED_SET = {"taa_btal_tqqq", "taa_btal_1n_tqqq", "taa_btal_lin_qqq"}
LONG_PROXY_DICT = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld", "taa_btal_1n_qld": "taa_1n_qld",
                   "taa_btal_lin_qqq": "taa_lin_qqq"}


def load_return_ser(alias_str: str, capital_float: float) -> pd.Series:
    """Daily returns from the day before the first position (NaN before)."""
    path_df = pd.read_csv(SOURCE_DIR_PATH / f"{alias_str}__{int(round(capital_float))}__path.csv.gz", index_col="date", parse_dates=True)
    nav_ser = path_df["total_value_float"].astype(float)
    invested_ser = path_df["portfolio_value_float"].abs() > 1e-9
    first_position_int = nav_ser.index.get_loc(invested_ser[invested_ser].index[0])
    return nav_ser.iloc[max(first_position_int - 1, 0):].pct_change()


STITCH_CUT_TS = pd.Timestamp("2012-10-02")


def stitched_ser(actual_ser: pd.Series, proxy_ser: pd.Series) -> pd.Series:
    """*** CRITICAL*** the stand-in fills ONLY the dates before the real sleeve exists (2012-10-02);
    from then on the real sleeve's own returns are used."""
    return pd.concat([proxy_ser[proxy_ser.index < STITCH_CUT_TS], actual_ser[actual_ser.index >= STITCH_CUT_TS]]).sort_index()


def activity_dict(alias_str: str, capital_float: float, start_ts: pd.Timestamp) -> dict:
    path_df = pd.read_csv(SOURCE_DIR_PATH / f"{alias_str}__{int(round(capital_float))}__path.csv.gz", index_col="date", parse_dates=True).loc[start_ts:]
    transaction_df = pd.read_csv(SOURCE_DIR_PATH / f"{alias_str}__{int(round(capital_float))}__transactions.csv.gz", parse_dates=["date"])
    transaction_df = transaction_df[transaction_df["date"] >= start_ts]
    year_count_float = (path_df.index[-1] - path_df.index[0]).days / 365.25
    return {
        "trade_days_per_year_float": transaction_df["date"].nunique() / year_count_float,
        "orders_per_year_float": len(transaction_df) / year_count_float,
        "commission_pct_per_year_float": float(transaction_df["commission_float"].sum() / path_df["total_value_float"].mean() / year_count_float),
        "mean_cash_weight_float": float((path_df["cash_float"].clip(lower=0) / path_df["total_value_float"]).mean()),
    }


def main() -> int:
    plan_dict = yaml.safe_load((HERE_PATH / "plan_frozen.yaml").read_text(encoding="utf-8"))
    capital_float = float(plan_dict["account"]["usd_capital_float"])
    reference_float = float(plan_dict["execution_realism"]["reference_capital_usd"])
    taa_capital_list = plan_dict["execution_realism"]["pod_capitals_usd"]["taa"]
    core5_capital_list = plan_dict["execution_realism"]["pod_capitals_usd"]["core5"]
    share_list = plan_dict["candidates"]["core5_capital_share_list"]
    w1_start_ts = pd.Timestamp(plan_dict["windows"]["w1"]["start_str"])
    w2_start_ts = pd.Timestamp(plan_dict["windows"]["w2"]["start_str"])
    end_ts = pd.Timestamp(plan_dict["end_date_str"])

    returns_dict = {}
    for alias_str in plan_dict["candidates"]["taa_variants"]:
        for c in taa_capital_list + [reference_float]:
            returns_dict[(alias_str, c)] = load_return_ser(alias_str, c)
    for c in core5_capital_list + [reference_float]:
        returns_dict[("core5", c)] = load_return_ser("core5", c)
    session_index = pd.DatetimeIndex(sorted(set().union(*[s.index for s in returns_dict.values()]))).sort_values()
    session_index = session_index[session_index <= end_ts]
    bench_df = common.build_benchmark_return_df(session_index, session_index[0].date().isoformat(), end_ts.date().isoformat())
    spy_ser = common.load_total_return_close_ser("SPY", session_index[0].date().isoformat(), end_ts.date().isoformat()).reindex(session_index).pct_change(fill_method=None)
    bench_df["SPY"] = spy_ser
    from data.norgate_loader import load_price_timeseries  # noqa: E402
    usdils_df = load_price_timeseries("USDILS", start_date_str=session_index[0].date().isoformat(), end_date_str=end_ts.date().isoformat())
    usdils_ser = usdils_df["Close"].astype(float)
    usdils_ser.index = pd.to_datetime(usdils_ser.index).normalize()
    usdils_ser = usdils_ser.reindex(session_index).ffill()
    fx_return_ser = usdils_ser.pct_change(fill_method=None)

    def window_ser(ser: pd.Series, start_ts: pd.Timestamp) -> tuple[pd.Series, pd.Timestamp]:
        sliced_ser = ser.reindex(session_index).loc[start_ts:end_ts]
        base_ts = session_index[session_index.get_loc(sliced_ser.index[0]) - 1]
        return sliced_ser, base_ts

    def metrics(ser: pd.Series, start_ts: pd.Timestamp) -> dict:
        sliced_ser, base_ts = window_ser(ser, start_ts)
        if sliced_ser.isna().any():
            return {}
        out = common.metric_dict(sliced_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
        nav_ser = common.nav_from_return_ser(sliced_ser)
        rolling_ser = (nav_ser / nav_ser.shift(252) - 1).dropna()
        out["worst_12m_float"] = float(rolling_ser.min())
        out["share_12m_negative_float"] = float((rolling_ser < 0).mean())
        return out

    # A. Variants standalone at the full account size, and the small-account drag.
    variant_row_list = []
    for alias_str in plan_dict["candidates"]["taa_variants"]:
        small_w1 = metrics(returns_dict[(alias_str, capital_float)], w1_start_ts)
        big_w1 = metrics(returns_dict[(alias_str, reference_float)], w1_start_ts)
        small_w2 = metrics(returns_dict[(alias_str, capital_float)], w2_start_ts)
        variant_row_list.append({
            "alias_str": alias_str, "wired_bool": alias_str in WIRED_SET,
            "cagr_w1": small_w1["cagr_float"], "vol_w1": small_w1["volatility_float"], "sharpe_w1": small_w1["sharpe_rf0_float"],
            "maxdd_w1": small_w1["max_drawdown_float"], "worst12m_w1": small_w1["worst_12m_float"], "beta_w1": small_w1["beta_spx_float"],
            "small_account_drag_w1": big_w1["cagr_float"] - small_w1["cagr_float"],
            "cagr_w2": small_w2.get("cagr_float", np.nan), "maxdd_w2": small_w2.get("max_drawdown_float", np.nan),
            **activity_dict(alias_str, capital_float, w1_start_ts),
        })
    variant_df = pd.DataFrame(variant_row_list).set_index("alias_str")

    # B. Variant rule.
    eligible_df = variant_df[variant_df["wired_bool"] & (variant_df["small_account_drag_w1"] <= 0.010) & (variant_df["maxdd_w1"] >= -0.25)]
    chosen_variant_str = eligible_df["cagr_w1"].idxmax() if len(eligible_df) else None

    # C. Mixes with CORE5 (annual reset; each pod at its real dollar size).
    mix_row_list, mix_series_dict = [], {}
    for alias_str in sorted(WIRED_SET):
        for share_float, taa_c, core5_c in zip(share_list, taa_capital_list, [None] + core5_capital_list[1:]):
            if share_float == 0.0:
                book_ser = returns_dict[(alias_str, capital_float)]
                long_book_ser = stitched_ser(returns_dict[(alias_str, capital_float)], returns_dict[(LONG_PROXY_DICT[alias_str], capital_float)])
            else:
                pair_df = pd.DataFrame({"taa": returns_dict[(alias_str, taa_c)], "core5": returns_dict[("core5", core5_c)]}).reindex(session_index)
                start_ts = pair_df.dropna().index[0]
                book_ser, _ = common.book_return_ser(pair_df.loc[start_ts:], {"taa": 1 - share_float, "core5": share_float}, "annual")
                long_pair_df = pd.DataFrame({"taa": stitched_ser(returns_dict[(alias_str, taa_c)], returns_dict[(LONG_PROXY_DICT[alias_str], taa_c)]),
                                             "core5": returns_dict[("core5", core5_c)]}).reindex(session_index)
                long_start_ts = long_pair_df.dropna().index[0]
                long_book_ser, _ = common.book_return_ser(long_pair_df.loc[long_start_ts:], {"taa": 1 - share_float, "core5": share_float}, "annual")
            w1 = metrics(book_ser, w1_start_ts)
            w2 = metrics(long_book_ser, w2_start_ts)
            mix_series_dict[(alias_str, share_float)] = (book_ser, long_book_ser)
            mix_row_list.append({"alias_str": alias_str, "core5_share": share_float, "core5_pod_usd": 0 if share_float == 0 else core5_c,
                                 "cagr_w1": w1["cagr_float"], "vol_w1": w1["volatility_float"], "sharpe_w1": w1["sharpe_rf0_float"],
                                 "maxdd_w1": w1["max_drawdown_float"], "worst12m_w1": w1["worst_12m_float"], "beta_w1": w1["beta_spx_float"],
                                 "cagr_w2_proxy": w2.get("cagr_float", np.nan), "maxdd_w2_proxy": w2.get("max_drawdown_float", np.nan),
                                 "worst12m_w2_proxy": w2.get("worst_12m_float", np.nan)})
    mix_df = pd.DataFrame(mix_row_list)

    # D. CORE5 rule for the chosen variant.
    chosen_mix_share = 0.0
    if chosen_variant_str:
        own_df = mix_df[mix_df["alias_str"] == chosen_variant_str].set_index("core5_share")
        base_row = own_df.loc[0.0]
        for share_float in [s for s in share_list if s > 0]:
            row = own_df.loc[share_float]
            if row["core5_pod_usd"] < 8000:
                continue
            if row["maxdd_w1"] >= 0.75 * base_row["maxdd_w1"] and base_row["cagr_w1"] - row["cagr_w1"] <= 0.03:
                chosen_mix_share = share_float
                break

    # E. Benchmarks and the ILS view.
    bench_row_list = []
    for label_str, ser in (("SPY buy & hold", bench_df["SPY"]), ("QQQ buy & hold", bench_df["QQQ"]), ("60/40", bench_df["SIXTY_FORTY"]),
                           ("T-bills", bench_df["TBILL"]), ("CORE5 alone @ $26.6K", returns_dict[("core5", capital_float)])):
        w1 = metrics(ser, w1_start_ts)
        w2 = metrics(ser, w2_start_ts)
        bench_row_list.append({"series_str": label_str, "cagr_w1": w1["cagr_float"], "vol_w1": w1["volatility_float"], "sharpe_w1": w1["sharpe_rf0_float"],
                               "maxdd_w1": w1["max_drawdown_float"], "worst12m_w1": w1["worst_12m_float"],
                               "cagr_w2": w2.get("cagr_float", np.nan), "maxdd_w2": w2.get("max_drawdown_float", np.nan)})
    bench_out_df = pd.DataFrame(bench_row_list)

    ils_row_list = []
    for label_str, ser in [(f"{chosen_variant_str} alone", returns_dict[(chosen_variant_str, capital_float)])] + (
            [(f"{chosen_variant_str} + CORE5 {chosen_mix_share:.0%}", mix_series_dict[(chosen_variant_str, chosen_mix_share)][0])] if chosen_mix_share > 0 else []) + [
            ("QQQ buy & hold", bench_df["QQQ"]), ("SPY buy & hold", bench_df["SPY"])]:
        ils_ser = (1 + ser.reindex(session_index)) * (1 + fx_return_ser) - 1
        w1 = metrics(ils_ser, w1_start_ts)
        ils_row_list.append({"series_str": label_str, "cagr_ils_w1": w1["cagr_float"], "vol_ils_w1": w1["volatility_float"],
                             "maxdd_ils_w1": w1["max_drawdown_float"], "worst12m_ils_w1": w1["worst_12m_float"]})
    usdils_change_float = float(usdils_ser.loc[w1_start_ts:].iloc[-1] / usdils_ser.loc[:w1_start_ts].iloc[-2] - 1)

    # F. Crises (long window, proxy before 2012-10 where needed).
    episode_list = [e for e in common.equity_drawdown_episode_list(bench_df["SPXTR"].loc[w2_start_ts:], -0.10)]
    crisis_row_list = []
    candidates_dict = {f"{chosen_variant_str} (stand-in before 2012-10)": mix_series_dict[(chosen_variant_str, 0.0)][1]}
    if chosen_mix_share > 0:
        candidates_dict[f"+ CORE5 {chosen_mix_share:.0%} (2008 via proxy)"] = mix_series_dict[(chosen_variant_str, chosen_mix_share)][1]
    for episode_dict in episode_list:
        row = {"episode": f"{episode_dict['peak_date_str']} to {episode_dict['trough_date_str']}", "spx": episode_dict["spx_drawdown_float"]}
        for label_str, ser in {**candidates_dict, "QQQ": bench_df["QQQ"], "60/40": bench_df["SIXTY_FORTY"]}.items():
            row[label_str] = common.window_return_float(ser.reindex(session_index).fillna(0.0), episode_dict["peak_date_str"], episode_dict["trough_date_str"])
        crisis_row_list.append(row)
    crisis_df = pd.DataFrame(crisis_row_list)

    # CORE5 small-pod executability.
    core5_scale_df = pd.DataFrame([{"core5_pod_usd": c, **{k: v for k, v in metrics(returns_dict[("core5", c)], w1_start_ts).items()
                                                            if k in ("cagr_float", "volatility_float", "max_drawdown_float")}}
                                   for c in core5_capital_list[1:] + [capital_float, reference_float]])

    for name_str, frame_df in (("variants", variant_df), ("mixes", mix_df), ("benchmarks", bench_out_df), ("ils", pd.DataFrame(ils_row_list)),
                               ("crises", crisis_df), ("core5_scale", core5_scale_df)):
        frame_df.to_csv(STUDY_DIR_PATH / f"{name_str}.csv", float_format="%.6g")
    verdict_dict = {"chosen_variant_str": chosen_variant_str, "chosen_core5_share_float": chosen_mix_share,
                    "eligible_variants": list(eligible_df.index), "usdils_change_w1_float": usdils_change_float}
    (STUDY_DIR_PATH / "verdict.json").write_text(json.dumps(verdict_dict, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(variant_df.round(3).to_string())
    print(mix_df.round(3).to_string())
    print(bench_out_df.round(3).to_string())
    print(pd.DataFrame(ils_row_list).round(3).to_string(), "\nUSDILS change over w1:", round(usdils_change_float, 3))
    print(crisis_df.round(3).to_string())
    print(core5_scale_df.round(4).to_string())
    print(json.dumps(verdict_dict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
