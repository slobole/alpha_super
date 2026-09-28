"""Strategy readiness audit 2026-09-28, HPI (both S&P 500 pods): synthetic pins of the audited mechanics.

These tests pin behaviour the audit found or relied on; they do not change production code.
Real-data evidence lives in results/research/strategy_readiness_audit_20260928/hpi/.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.hpi.stateful_long import (  # noqa: E402
    ENTRY_HORIZON_VOTE_STR,
    TURNOVER_FIELD_STR,
    HPIStatefulLongStrategy,
    compute_strict_hpi,
)

LIVE_MARKER_FLOAT = 1.0  # alpha/live/strategy_host.py HPI_LIVE_TRADABLE_OPEN_MARKER_FLOAT


def make_strategy(max_positions_int=2, entry_mode_str="baseline", **kwargs):
    return HPIStatefulLongStrategy(
        name="HPIAuditTest", benchmarks=[], ranking_field_str=TURNOVER_FIELD_STR, capital_base=100_000.0,
        slippage=0.0, commission_per_share=0.0, commission_minimum=0.0, max_positions_int=max_positions_int,
        entry_mode_str=entry_mode_str, **kwargs)


def row(values):
    ser = pd.Series(values, dtype=float)
    ser.index = pd.MultiIndex.from_tuples(ser.index)
    return ser


def eligible(sym, turnover):
    return {(sym, "Close"): 105.0, (sym, TURNOVER_FIELD_STR): turnover, (sym, "return_3d_ser"): -0.03,
            (sym, "hpi_value_ser"): 20.0, (sym, "sma_200_price_ser"): 100.0, (sym, "ibs_value_ser"): 0.05,
            (sym, "rsi2_value_ser"): 20.0}


def held(sym, ibs=0.5):
    return {(sym, "Close"): 100.0, (sym, TURNOVER_FIELD_STR): 1.0, (sym, "return_3d_ser"): 0.01,
            (sym, "hpi_value_ser"): 60.0, (sym, "sma_200_price_ser"): 90.0, (sym, "ibs_value_ser"): ibs,
            (sym, "rsi2_value_ser"): 50.0}


def synthetic_prices(n=1_500, seed=7, symbols=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    cols = {}
    for i, s in enumerate(symbols):
        close = 50.0 * (1 + i) * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
        high = close * (1 + rng.uniform(0, 0.02, n))
        low = close * (1 - rng.uniform(0, 0.02, n))
        opn = low + (high - low) * rng.uniform(0, 1, n)
        vol = rng.uniform(1e6, 5e6, n)
        cols.update({(s, "Open"): opn, (s, "High"): high, (s, "Low"): low, (s, "Close"): close,
                     (s, "Volume"): vol, (s, "Turnover"): close * vol, (s, "Unadjusted Close"): close * 2.0,
                     (s, "Dividend"): np.zeros(n)})
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


FEATURES = ("return_3d_ser", "hpi_value_ser", "sma_200_price_ser", "ibs_value_ser", "rsi2_value_ser",
            "return_2d_ser", "return_5d_ser", "hpi_2d_ser", "hpi_5d_ser")


# ---------------------------------------------------------------------------------------------- A1 / A3


def test_strict_hpi_matches_bruteforce_definition():
    rng = np.random.default_rng(3)
    ret = pd.Series(np.round(rng.normal(0, 0.03, 200), 3))  # rounding creates ties
    lookback = 20
    got = compute_strict_hpi(ret, lookback_int=lookback)
    for t in range(lookback, len(ret)):
        prior = ret.iloc[t - lookback:t].to_numpy()
        x = ret.iloc[t]
        if x <= 0:
            den = (prior <= 0).sum()
            exp = np.nan if den == 0 else 100.0 * (prior <= x).sum() / den
        else:
            den = (prior > 0).sum()
            exp = np.nan if den == 0 else 100.0 * (prior > x).sum() / den
        if np.isnan(exp):
            assert np.isnan(got.iloc[t])
        else:
            assert got.iloc[t] == pytest.approx(exp, abs=1e-9)


def test_features_row_T_are_truncation_invariant():
    prices = synthetic_prices()
    strat = make_strategy(entry_mode_str=ENTRY_HORIZON_VOTE_STR)
    full = strat.compute_signals(prices)
    for cut in (prices.index[1_300], prices.index[1_420], prices.index[-2]):
        trunc = strat.compute_signals(prices.loc[:cut])
        for s in ("AAA", "BBB", "CCC"):
            for f in FEATURES:
                a, b = full.loc[cut, (s, f)], trunc.loc[cut, (s, f)]
                assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-12), (cut, s, f)


def test_positive_control_one_bar_leak_is_caught_by_row_T_truncation():
    prices = synthetic_prices()

    class Leaky(HPIStatefulLongStrategy):
        def compute_signals(self, pricing_data_df):
            out = super().compute_signals(pricing_data_df)
            cols = [c for c in out.columns if c[1] == "ibs_value_ser"]
            out[cols] = out[cols].shift(-1)
            return out

    strat = Leaky(name="leak", benchmarks=[], ranking_field_str=TURNOVER_FIELD_STR)
    full = strat.compute_signals(prices)
    cut = prices.index[1_400]
    trunc = strat.compute_signals(prices.loc[:cut])
    assert not np.isnan(full.loc[cut, ("AAA", "ibs_value_ser")])
    assert np.isnan(trunc.loc[cut, ("AAA", "ibs_value_ser")])  # the harness difference that flags the leak


# ---------------------------------------------------------------------------------------------- A2


@pytest.mark.parametrize("k", [40.0, 0.1, 1.5])
def test_features_and_ranking_invariant_to_future_split(k):
    prices = synthetic_prices()
    scaled = prices.copy()
    for f in ("Open", "High", "Low", "Close", "Dividend"):
        scaled[("BBB", f)] = scaled[("BBB", f)] / k
    scaled[("BBB", "Volume")] = scaled[("BBB", "Volume")] * k  # Turnover and Unadjusted Close stay nominal
    strat = make_strategy(entry_mode_str=ENTRY_HORIZON_VOTE_STR)
    a, b = strat.compute_signals(prices), strat.compute_signals(scaled)
    scale_free = [f for f in FEATURES if f != "sma_200_price_ser"] + [TURNOVER_FIELD_STR]
    # SMA200 is a price level; the decision uses Close_T > SMA200_T, i.e. the scale-free ratio.
    a[("BBB", "close_over_sma")] = a[("BBB", "Close")] / a[("BBB", "sma_200_price_ser")]
    b[("BBB", "close_over_sma")] = b[("BBB", "Close")] / b[("BBB", "sma_200_price_ser")]
    for f in scale_free + ["close_over_sma"]:
        x, y = a[("BBB", f)].to_numpy(dtype=float), b[("BBB", f)].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(x), np.isnan(y))
        np.testing.assert_allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------------------------- B / G-033


def _state_with_pending_exit(strat, idx):
    strat.universe_df = pd.DataFrame({"OLD": [1, 1], "NEW": [1, 1]}, index=idx)
    strat.add_transaction(7, idx[0], "OLD", 10, 100.0, 1_000.0, 1, 0.0)
    strat.current_trade_map["OLD"] = 7
    strat.previous_bar, strat.current_bar = idx[0], idx[1]


def test_backtest_keeps_slot_when_pending_exit_open_missing_but_live_marker_refills():
    idx = pd.bdate_range("2024-03-07", periods=2)
    signal = row({**held("OLD", ibs=0.95), **eligible("NEW", 50.0)})

    backtest = make_strategy(max_positions_int=1)
    _state_with_pending_exit(backtest, idx)
    backtest.iterate(pd.DataFrame(index=idx[:1]), signal, pd.Series({"OLD": np.nan}))
    assert backtest.get_orders() == []  # no exit order, no refill: slot kept (knows Open_(T+1) is missing)

    live = make_strategy(max_positions_int=1)
    _state_with_pending_exit(live, idx)
    live.iterate(pd.DataFrame(index=idx[:1]), signal, pd.Series({"OLD": LIVE_MARKER_FLOAT}))
    assets = [(o.asset, o.target) for o in live.get_orders()]
    assert assets == [("OLD", True), ("NEW", False)]  # exit + same-open refill


def test_held_name_with_no_bar_on_T_exits_through_membership_zero():
    """Norgate index_constituent_timeseries has no row on a no-bar day, so the concatenated universe is 0 there
    (audit: 0 of 10,979 real no-bar member-lifetime days carried membership 1). A halted held name is therefore
    treated as removed: exit + refill (live), liquidation at the last close if Open_(T+1) is also missing (backtest).
    """
    idx = pd.bdate_range("2024-03-07", periods=2)
    strat = make_strategy(max_positions_int=1)
    strat.universe_df = pd.DataFrame({"OLD": [0, 1], "NEW": [1, 1]}, index=idx)  # OLD has no bar on T
    strat.add_transaction(7, idx[0] - pd.Timedelta(days=1), "OLD", 10, 100.0, 1_000.0, 1, 0.0)
    strat.current_trade_map["OLD"] = 7
    strat.previous_bar, strat.current_bar = idx[0], idx[1]
    signal = row({(("OLD", "Close")): 100.0, **eligible("NEW", 50.0)})  # OLD features NaN on a no-bar day
    strat.iterate(pd.DataFrame(index=idx[:1]), signal, pd.Series({"OLD": LIVE_MARKER_FLOAT}))
    assert [(o.asset, o.target) for o in strat.get_orders()] == [("OLD", True), ("NEW", False)]


def test_entry_value_is_one_tenth_of_previous_total_value():
    idx = pd.bdate_range("2024-03-07", periods=2)
    strat = make_strategy(max_positions_int=10)
    strat.universe_df = pd.DataFrame({"NEW": [1, 1]}, index=idx)
    strat.previous_bar, strat.current_bar = idx[0], idx[1]
    strat._total_value_history_list = [123_456.0]
    strat.iterate(pd.DataFrame(index=idx[:1]), row(eligible("NEW", 5.0)), pd.Series(dtype=float))
    (order,) = strat.get_orders()
    assert order.unit == "value" and order.amount == pytest.approx(12_345.6)


# ---------------------------------------------------------------------------------------------- A8


def test_removal_liquidation_fee_ignores_historical_share_units():
    """stateful_long.py:673 uses _compute_commission (adjusted units) where the base engine uses
    _compute_execution_commission_float (raw-equivalent units when historical_share_units_bool is on)."""
    idx = pd.bdate_range("2024-03-07", periods=2)
    strat = HPIStatefulLongStrategy(name="fee", benchmarks=[], ranking_field_str=TURNOVER_FIELD_STR,
                                    commission_per_share=0.005, commission_minimum=1.0)
    strat.historical_share_units_bool = True
    strat.universe_df = pd.DataFrame({"OLD": [1, 0]}, index=idx)
    strat.add_transaction(7, idx[0], "OLD", 4_000.0, 25.0, 100_000.0, 1, 0.0)  # adjusted units
    prices = pd.DataFrame({("OLD", "Open"): [25.0, np.nan], ("OLD", "Close"): [25.0, 25.0],
                           ("OLD", "Unadjusted Close"): [100.0, 100.0]}, index=idx)  # later 4:1 split
    prices.columns = pd.MultiIndex.from_tuples(prices.columns)
    strat.previous_bar, strat.current_bar = idx[0], idx[1]
    _, fee = strat._liquidate_missing_price_positions(prices)
    assert fee == pytest.approx(20.0)  # 4,000 adjusted shares x $0.005
    assert strat._compute_execution_commission_float(prices, "OLD", -4_000.0, idx[0]) == pytest.approx(5.0)
