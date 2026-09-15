"""Frozen QQQ Golden Cross baseline versus a matched buy-and-hold account.

Run after plan.json, data_manifest.json and prices.parquet have been frozen:
    python -m scripts.research.run_qqq_golden_cross_baseline

No data download, parameter search, engine changes or LIVE operations occur here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from strategies.momentum.strategy_mo_qqq_golden_cross import QqqGoldenCrossStrategy

ROOT_PATH = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT_PATH / "results/research/qqq_golden_cross_baseline_20260915"


class QqqBuyHoldComparison(QqqGoldenCrossStrategy):
    """Same accounting, one purchase after the shared first decision close."""

    def __init__(self, capital_base_float: float):
        super().__init__(capital_base_float)
        self.name = "qqq_buy_hold_matched"
        self.entry_attempted_bool = False

    def iterate(self, data_df, close_row_ser, open_price_ser):
        if self.previous_bar < self.decision_start_ts or self.entry_attempted_bool:
            return
        # *** CRITICAL*** Fixed at the initial decision close, not the fill open.
        share_int = int(np.floor(self.previous_total_value / float(close_row_ser[("QQQ", "Close")])))
        self.entry_attempted_bool = True
        if share_int > 0:
            self.trade_id_int = 1
            self.order_target("QQQ", share_int, trade_id=1)


def independent_account(
    pricing_df: pd.DataFrame, start_position_int: int, capital_float: float, buy_hold_bool: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scalar oracle: independent rolling sums, decisions, fills and bookkeeping."""
    close_vec = pricing_df[("QQQ", "Close")].to_numpy(dtype=float)
    open_vec = pricing_df[("QQQ", "Open")].to_numpy(dtype=float)
    dividend_vec = pricing_df[("QQQ", "Dividend")].to_numpy(dtype=float)
    share_int, cash_float, nav_float = 0, capital_float, capital_float
    daily_list, fill_list = [], []
    for position_int in range(start_position_int, len(pricing_df)):
        previous_nav_float = nav_float
        # *** CRITICAL*** Dividend[T] is entitlement at Close_T; only shares
        # held before Open_(T+1) earn it. Cash changes before that open's orders.
        gross_dividend_float = share_int * dividend_vec[position_int - 1]
        cash_float += gross_dividend_float - max(gross_dividend_float, 0.0) * 0.25
        order_share_int = 0
        if position_int > start_position_int:
            decision_position_int = position_int - 1
            if buy_hold_bool:
                buy_bool = position_int == start_position_int + 1
                sell_bool = False
            else:
                # *** CRITICAL*** Both inclusive windows end no later than T.
                spread_float = (
                    sum(close_vec[decision_position_int - 49:decision_position_int + 1]) / 50
                    - sum(close_vec[decision_position_int - 199:decision_position_int + 1]) / 200
                )
                prior_spread_float = (
                    sum(close_vec[decision_position_int - 50:decision_position_int]) / 50
                    - sum(close_vec[decision_position_int - 200:decision_position_int]) / 200
                )
                buy_bool = prior_spread_float <= 0 < spread_float
                sell_bool = prior_spread_float >= 0 > spread_float
            if share_int == 0 and buy_bool:
                order_share_int = int(np.floor(previous_nav_float / close_vec[decision_position_int]))
            elif share_int > 0 and sell_bool:
                order_share_int = -share_int
        if order_share_int:
            fill_price_float = open_vec[position_int] * (1 + np.sign(order_share_int) * 0.00025)
            fee_float = max(1.0, abs(order_share_int) * 0.005)
            cash_float -= order_share_int * fill_price_float + fee_float
            share_int += order_share_int
            fill_list.append({
                "bar": pricing_df.index[position_int], "amount": order_share_int,
                "price": fill_price_float, "commission": fee_float,
            })
        nav_float = cash_float + share_int * close_vec[position_int]
        daily_list.append({
            "date": pricing_df.index[position_int], "cash": cash_float,
            "shares": share_int, "total_value": nav_float,
        })
    return pd.DataFrame(daily_list).set_index("date"), pd.DataFrame(
        fill_list, columns=["bar", "amount", "price", "commission"]
    )


def verify_account(strategy_obj, pricing_df, start_position_int, buy_hold_bool):
    oracle_df, oracle_fill_df = independent_account(
        pricing_df, start_position_int, strategy_obj._capital_base, buy_hold_bool,
    )
    engine_df = strategy_obj.results.astype(float).copy()
    assert engine_df.index.equals(oracle_df.index)
    for field_str in ("cash", "total_value"):
        np.testing.assert_allclose(engine_df[field_str], oracle_df[field_str], rtol=1e-11, atol=1e-5)
    engine_share_ser = engine_df["portfolio_value"] / pricing_df.loc[engine_df.index, ("QQQ", "Close")]
    np.testing.assert_allclose(engine_share_ser, oracle_df["shares"], rtol=0, atol=1e-7)
    fill_df = strategy_obj.get_transactions()
    assert list(pd.to_datetime(fill_df["bar"])) == list(pd.to_datetime(oracle_fill_df["bar"]))
    for field_str in ("amount", "price", "commission"):
        np.testing.assert_allclose(fill_df[field_str].astype(float), oracle_fill_df[field_str], rtol=1e-11, atol=1e-7)
    return oracle_df, {
        "daily_rows": len(engine_df), "fills": len(fill_df),
        "max_cash_error_usd": float((engine_df["cash"] - oracle_df["cash"]).abs().max()),
        "max_nav_error_usd": float((engine_df["total_value"] - oracle_df["total_value"]).abs().max()),
        "passed": True,
    }


def performance_metrics(daily_df, anchor_nav_float, anchor_date_ts, drop_initial_bool=False):
    """CAGR calendar-years; sample-SD252 Sharpe; block boundary NAV retained."""
    nav_ser = daily_df["total_value"].astype(float)
    # *** CRITICAL*** Report-only previous NAV; block first return includes the
    # previous block's last close. No portfolio restart or omitted boundary day.
    previous_nav_ser = nav_ser.shift(1)
    previous_nav_ser.iloc[0] = anchor_nav_float
    return_ser = nav_ser / previous_nav_ser - 1
    if drop_initial_bool:
        return_ser = return_ser.iloc[1:]
    years_float = (nav_ser.index[-1] - pd.Timestamp(anchor_date_ts)).days / 365.25
    total_return_float = float(nav_ser.iloc[-1] / anchor_nav_float - 1)
    volatility_float = float(return_ser.std(ddof=1) * np.sqrt(252))
    sharpe_float = float(return_ser.mean() * 252 / volatility_float) if volatility_float > 0 else 0.0
    peak_vec = np.maximum.accumulate(np.r_[anchor_nav_float, nav_ser.to_numpy()])[1:]
    drawdown_ser = nav_ser / peak_vec - 1
    cagr_float = (1 + total_return_float) ** (1 / years_float) - 1
    return {
        "start": str(nav_ser.index[0].date()), "end": str(nav_ser.index[-1].date()),
        "final_nav": float(nav_ser.iloc[-1]), "total_return_pct": total_return_float * 100,
        "cagr_pct": cagr_float * 100, "volatility_pct": volatility_float * 100,
        "sharpe": sharpe_float, "max_drawdown_pct": float(drawdown_ser.min() * 100),
        "mar": cagr_float / abs(float(drawdown_ser.min())) if drawdown_ser.min() < 0 else 0.0,
        "close_held_exposure_pct": float(daily_df["portfolio_value"].gt(0).mean() * 100),
        "average_invested_nav_pct": float((daily_df["portfolio_value"] / nav_ser).mean() * 100),
    }


def account_details(strategy_obj, pricing_df, start_position_int):
    daily_df = strategy_obj.results.astype(float)
    fill_df = strategy_obj.get_transactions()
    years_float = (daily_df.index[-1] - daily_df.index[0]).days / 365.25
    previous_nav_ser = daily_df["total_value"].shift(1)
    previous_nav_ser.iloc[0] = strategy_obj._capital_base
    slippage_float, turnover_float = 0.0, 0.0
    for _, fill_ser in fill_df.iterrows():
        fill_ts = pd.Timestamp(fill_ser["bar"])
        reference_open_float = float(pricing_df.loc[fill_ts, ("QQQ", "Open")])
        slippage_float += abs(float(fill_ser["amount"])) * abs(float(fill_ser["price"]) - reference_open_float)
        turnover_float += abs(float(fill_ser["amount"]) * float(fill_ser["price"])) / previous_nav_ser.loc[fill_ts]
    policy_dict = strategy_obj._accounting_policy_dict
    first_entry_ts = pd.Timestamp(fill_df.iloc[0]["bar"]) if len(fill_df) else None
    return {
        "entry_count": int(fill_df["amount"].gt(0).sum()),
        "completed_trades": int(fill_df["amount"].lt(0).sum()),
        "first_entry": str(first_entry_ts.date()) if first_entry_ts is not None else None,
        "initial_cash_sessions": int(daily_df.index.get_loc(first_entry_ts)) if first_entry_ts is not None else len(daily_df),
        "final_shares": float(strategy_obj.get_position("QQQ")),
        "commission_usd": float(fill_df["commission"].sum()),
        "slippage_usd": float(slippage_float),
        "direct_trading_cost_usd": float(slippage_float + fill_df["commission"].sum()),
        "two_sided_turnover_ann_pct": float(turnover_float / years_float * 100),
        "net_dividends_usd": float(strategy_obj.dividend_cash_net_total_float),
        **{key_str: value_obj for key_str, value_obj in policy_dict.items()
           if key_str.startswith(("negative_cash_day", "negative_cash_episode", "minimum_cash", "average_negative_cash"))},
    }


def run_study():
    plan_path, manifest_path = OUTPUT_PATH / "plan.json", OUTPUT_PATH / "data_manifest.json"
    plan_dict = json.loads(plan_path.read_text())
    manifest_dict = json.loads(manifest_path.read_text())
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == manifest_dict["plan_sha256"]
    assert hashlib.sha256((OUTPUT_PATH / "prices.parquet").read_bytes()).hexdigest() == manifest_dict["pricing_sha256"]
    for source_str, expected_hash_str in manifest_dict["source_sha256"].items():
        assert hashlib.sha256((ROOT_PATH / source_str).read_bytes()).hexdigest() == expected_hash_str, source_str
    assert not (OUTPUT_PATH / "comparison.csv").exists(), "Results already exist; do not overwrite."
    pricing_df = pd.read_parquet(OUTPUT_PATH / "prices.parquet")
    pricing_df.attrs.update(manifest_dict["attrs"])
    start_position_int = 200
    capital_float = float(plan_dict["capital_usd"])
    result_row_list, subperiod_row_list, validation_dict, equity_dict = [], [], {}, {}
    for account_str, strategy_obj in (
        ("Golden Cross", QqqGoldenCrossStrategy(capital_float)),
        ("Buy and hold", QqqBuyHoldComparison(capital_float)),
    ):
        run_daily(
            strategy_obj, pricing_df, calendar=pricing_df.index[start_position_int:],
            show_progress=False, show_signal_progress_bool=False, audit_override_bool=None,
        )
        oracle_df, validation_dict[account_str] = verify_account(
            strategy_obj, pricing_df, start_position_int, account_str == "Buy and hold",
        )
        key_str = "golden_cross" if account_str == "Golden Cross" else "buy_hold"
        strategy_obj.results.to_csv(OUTPUT_PATH / f"{key_str}_daily.csv", index_label="date")
        strategy_obj.get_transactions().to_csv(OUTPUT_PATH / f"{key_str}_fills.csv", index=False)
        strategy_obj.get_dividend_ledger().to_csv(OUTPUT_PATH / f"{key_str}_dividends.csv", index=False)
        oracle_df.to_csv(OUTPUT_PATH / f"{key_str}_independent.csv")
        daily_df = strategy_obj.results.astype(float)
        equity_dict[account_str] = daily_df["total_value"]
        result_row_list.append({
            "account": account_str,
            **performance_metrics(daily_df, capital_float, daily_df.index[0], True),
            **account_details(strategy_obj, pricing_df, start_position_int),
        })
        for start_str, end_str in (
            ("1990-01-01", "2009-12-31"), ("2010-01-01", "2019-12-31"),
            ("2020-01-01", "2026-09-14"),
        ):
            block_df = daily_df.loc[start_str:end_str]
            first_position_int = int(daily_df.index.get_loc(block_df.index[0]))
            anchor_float = capital_float if first_position_int == 0 else daily_df["total_value"].iloc[first_position_int - 1]
            anchor_ts = daily_df.index[0 if first_position_int == 0 else first_position_int - 1]
            subperiod_row_list.append({
                "account": account_str,
                **performance_metrics(block_df, anchor_float, anchor_ts, first_position_int == 0),
            })
    comparison_df = pd.DataFrame(result_row_list).set_index("account")
    subperiod_df = pd.DataFrame(subperiod_row_list)
    comparison_df.to_csv(OUTPUT_PATH / "comparison.csv")
    subperiod_df.to_csv(OUTPUT_PATH / "subperiods.csv", index=False)
    validation_dict["study_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUTPUT_PATH / "verification.json").write_text(json.dumps(validation_dict, indent=2), encoding="utf-8")
    equity_df = pd.DataFrame(equity_dict)
    figure_obj, axis_vec = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for account_str, color_str in (("Golden Cross", "#176b87"), ("Buy and hold", "#b65f28")):
        nav_ser = equity_df[account_str]
        axis_vec[0].plot(nav_ser.index, nav_ser, label=account_str, color=color_str, linewidth=1.4)
        axis_vec[1].plot(nav_ser.index, (nav_ser / nav_ser.cummax() - 1) * 100, color=color_str, linewidth=1)
    axis_vec[0].set_yscale("log")
    axis_vec[0].set_ylabel("Account value USD (log scale)")
    axis_vec[0].set_title("QQQ: Golden Cross 50/200 vs matched buy and hold")
    axis_vec[0].legend()
    axis_vec[1].set_ylabel("Drawdown (%)")
    for axis_obj in axis_vec:
        axis_obj.grid(alpha=0.2)
    figure_obj.text(0.5, 0.015, "Net trading costs and 25% dividend withholding. Cash interest 0%; negative-cash financing unmodeled.", ha="center", fontsize=9)
    figure_obj.tight_layout(rect=(0, 0.035, 1, 1))
    figure_obj.savefig(OUTPUT_PATH / "comparison.png", dpi=160)
    plt.close(figure_obj)
    print(comparison_df.to_json(orient="index", indent=2))
    print(subperiod_df.to_json(orient="records", indent=2))
    print(json.dumps(validation_dict, indent=2))


if __name__ == "__main__":
    run_study()
