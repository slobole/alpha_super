"""Strategy readiness audit 2026-09-28, DV2 + QPI: synthetic pins of the audited mechanics.

These tests pin behaviour the audit found or relied on; they do not change production code.
Real-data evidence lives in results/research/strategy_readiness_audit_20260928/mr_dv2_qpi/.
"""

import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily  # noqa: E402
from alpha.indicators import qp_indicator, qp_indicator_reference  # noqa: E402
from alpha.live import strategy_host  # noqa: E402
from alpha.live.models import LiveRelease, PodState  # noqa: E402
from strategies.dv2 import strategy_mr_dv2 as dv2_mod  # noqa: E402
from strategies.qpi import strategy_mr_qpi_ibs_rsi_exit as qpi_mod  # noqa: E402

QPI_TEST_PARAMS = {"max_positions_int": 3, "qpi_lookback_years_int": 1, "sma_window_int": 20,
                   "qpi_threshold_float": 30.0, "qpi_window_int": 3, "return_lookback_days_int": 3,
                   "max_entry_ibs_float": 0.25, "exit_ibs_threshold_float": 0.90, "rsi_window_int": 2,
                   "exit_rsi2_threshold_float": 90.0}


# --------------------------------------------------------------------------- helpers
def xnys_sessions(n, start="2021-01-04"):
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6)))
    return pd.DatetimeIndex(sessions[:n]).tz_localize(None)


def synthetic_panel(n=700, symbols=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"), seed=11):
    rng = np.random.default_rng(seed)
    idx = xnys_sessions(n)
    cols = {}
    for i, s in enumerate(symbols):
        ret = rng.normal(0.0012, 0.022, n)
        close = (40.0 + 15 * i) * np.exp(np.cumsum(ret))
        high = close * (1 + rng.uniform(0.001, 0.03, n))
        low = close * (1 - rng.uniform(0.001, 0.03, n))
        opn = np.clip(close * (1 + rng.normal(0, 0.006, n)), low, high)
        vol = rng.uniform(1e6, 3e6, n)
        cols.update({(s, "Open"): opn, (s, "High"): high, (s, "Low"): low, (s, "Close"): close, (s, "Volume"): vol,
                     (s, "Turnover"): vol * close * (1 + 0.1 * i), (s, "Unadjusted Close"): close})
    spx = 3000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    cols.update({("$SPX", "Open"): spx, ("$SPX", "High"): spx, ("$SPX", "Low"): spx, ("$SPX", "Close"): spx})
    prices = pd.DataFrame(cols, index=idx)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns)
    universe = pd.DataFrame(1, index=idx, columns=list(symbols))
    universe.loc[idx[n // 2]:, symbols[-1]] = 0  # one name leaves the index half way
    return prices, universe


class _Recorder:
    def iterate(self, data, close, open_prices):
        if not hasattr(self, "decision_log"):
            self.decision_log = []
        if data is None or close is None:
            return super().iterate(data, close, open_prices)
        trade_map = dict(getattr(self, "current_trade_map", None) or getattr(self, "current_trade", {}) or {})
        rec = {"T": pd.Timestamp(self.previous_bar),
               "positions": {str(k): float(v) for k, v in self.get_positions().items() if v != 0},
               "cash": float(self.cash), "prev_total_value": float(self.previous_total_value),
               "trade_id": int(getattr(self, "trade_id_int", getattr(self, "trade_id", 0))),
               "current_trade_map": {str(k): int(v) for k, v in trade_map.items()}}
        super().iterate(data, close, open_prices)
        rec["orders"] = [(str(o.asset), str(o.unit), bool(o.target), float(o.amount)) for o in self.get_orders()]
        self.decision_log.append(rec)


class RecDV2(_Recorder, dv2_mod.DVO2Strategy):
    pass


class RecQPI(_Recorder, qpi_mod.QPIIbsRsiExitStrategy):
    pass


def run_backtest(fam, prices, universe):
    if fam == "dv2":
        s = RecDV2(name="t", benchmarks=["$SPX"], capital_base=100_000.0, slippage=0.00025,
                   commission_per_share=0.005, commission_minimum=1.0)
        s.max_positions = 3
        s.trade_id = 0
        s.current_trade = defaultdict(lambda: -1)
    else:
        s = RecQPI(name="t", benchmarks=["$SPX"], capital_base=100_000.0, **QPI_TEST_PARAMS)
    s.universe_df = universe
    calendar = prices.index[260:]
    run_daily(s, prices, calendar, show_progress=False, show_signal_progress_bool=False)
    return s


def release(fam):
    params = {"benchmark_list_str": ["$SPX"], "start_date_str": "1998-01-01", "indexname_str": "S&P 500",
              "max_positions_int": 3}
    if fam == "qpi":
        params.update(QPI_TEST_PARAMS)
    imp = ("strategies.dv2.strategy_mr_dv2:DVO2Strategy" if fam == "dv2"
           else "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy")
    return LiveRelease(release_id_str="audit", user_id_str="audit", pod_id_str="pod_audit", account_route_str="DU1",
                       strategy_import_str=imp, mode_str="paper", session_calendar_id_str="XNYS",
                       signal_clock_str="eod_snapshot_ready", execution_policy_str="next_open_moo",
                       data_profile_str="norgate_eod_sp500_pit", params_dict=params, risk_profile_str="x",
                       enabled_bool=True, source_path_str="<test>")


def patch_loaders(monkeypatch, mod, prices, universe, asof):
    monkeypatch.setattr(mod, "build_index_constituent_matrix",
                        lambda indexname="S&P 500": (list(universe.columns), universe.loc[:asof]))

    def get_prices(symbols, benchmarks, *args, **kwargs):
        end = kwargs.get("end_date", kwargs.get("end_date_str"))
        want = set(symbols) | set(benchmarks)
        return prices.loc[: pd.Timestamp(end), [c for c in prices.columns if c[0] in want]]

    monkeypatch.setattr(mod, "get_prices", get_prices)


def pod_state(rec):
    return PodState(pod_id_str="pod_audit", user_id_str="audit", account_route_str="DU1",
                    position_amount_map=dict(rec["positions"]), cash_float=rec["cash"],
                    total_value_float=rec["prev_total_value"],
                    strategy_state_dict={"trade_id_int": rec["trade_id"], "current_trade_map": rec["current_trade_map"]},
                    updated_timestamp_ts=datetime(2020, 1, 1))


# --------------------------------------------------------------------------- QPI indicator
def test_qpi_fast_matches_reference_with_ties_and_gaps():
    rng = np.random.default_rng(3)
    close = pd.Series(np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900))), 1))  # rounding creates ties
    close.iloc[[5, 400, 401]] = np.nan
    fast = qp_indicator(close, window_int=3, lookback_years_int=1)
    ref = qp_indicator_reference(close, window_int=3, lookback_years_int=1)
    assert (fast.isna() == ref.isna()).all()
    np.testing.assert_allclose(fast.dropna(), ref.dropna(), rtol=0, atol=1e-10)


def test_qpi_window_includes_current_bar():
    # 251 positive 3-day returns, then the current return is the unique minimum and the only down return.
    close = pd.Series(np.concatenate([100 * 1.001 ** np.arange(254), [90.0]]))
    q = qp_indicator(close, window_int=3, lookback_years_int=1)
    L = 252
    # rank of the current return inside its own window = 1/L; p_down = 1/L (only the current return is <= 0)
    assert q.iloc[-1] == pytest.approx(100.0 * (1.0 / L) / (1.0 / L))


def test_qpi_is_truncation_invariant_and_one_bar_leak_is_caught():
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 800))))
    full = qp_indicator(close, lookback_years_int=1)
    for t in (300, 555, 799):
        assert full.iloc[: t + 1].equals(qp_indicator(close.iloc[: t + 1], lookback_years_int=1))
    leak_full = qp_indicator(close.shift(-1), lookback_years_int=1)
    leak_trunc = qp_indicator(close.iloc[:556].shift(-1), lookback_years_int=1)
    assert not (leak_full.iloc[555] == leak_trunc.iloc[555])  # NaN vs value: the row-T harness catches it


@pytest.mark.parametrize("k", [40.0, 0.1, 1.5])
def test_qpi_is_invariant_to_future_split(k):
    rng = np.random.default_rng(9)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 700))))
    np.testing.assert_allclose(qp_indicator(close / k, lookback_years_int=1).dropna(),
                               qp_indicator(close, lookback_years_int=1).dropna(), rtol=0, atol=1e-9)


def test_documents_risk_one_missing_close_disables_qpi_for_a_full_lookback():
    close = pd.Series(100 * 1.0005 ** np.arange(900) * (1 + 0.01 * np.sin(np.arange(900))))
    close.iloc[500] = np.nan
    q = qp_indicator(close, window_int=3, lookback_years_int=1)
    # every window that contains one of the 4 NaN 3-day returns is NaN: 252 + 3 sessions after the gap
    assert q.iloc[500:500 + 252 + 3].isna().all()
    assert q.iloc[500 + 252 + 3:].notna().all()


# --------------------------------------------------------------------------- live host replay (B1)
# QPI lost its live route on 2026-09-28 (commit 40675e9), so the live-host replay
# tests run for DV2 only; the QPI backtest-side tests above still apply.
@pytest.mark.parametrize("fam", ["dv2"])
def test_live_host_reproduces_backtest_orders_on_every_decision(monkeypatch, fam):
    prices, universe = synthetic_panel()
    bt = run_backtest(fam, prices, universe)
    mod = dv2_mod
    run_fn = strategy_host._run_dv2_strategy_for_live_decision
    with_orders = [r for r in bt.decision_log if r["orders"]]
    assert len(with_orders) >= 20
    assert any(any(o[2] for o in r["orders"]) and any(not o[2] for o in r["orders"]) for r in with_orders)
    for rec in with_orders + bt.decision_log[::25]:
        T = rec["T"]
        patch_loaders(monkeypatch, mod, prices, universe, T)
        signal_ts, live = run_fn(release(fam), datetime(T.year, T.month, T.day, 20), pod_state(rec))
        assert pd.Timestamp(signal_ts) == T
        live_orders = [(str(o.asset), str(o.unit), bool(o.target), float(o.amount)) for o in live.get_orders()]
        assert live_orders == rec["orders"], T


@pytest.mark.parametrize("fam", ["dv2"])
def test_live_plan_entry_weight_is_exactly_one_over_max_positions(monkeypatch, fam):
    prices, universe = synthetic_panel()
    bt = run_backtest(fam, prices, universe)
    rec = next(r for r in bt.decision_log if any(not o[2] for o in r["orders"]))
    mod = dv2_mod
    builder = strategy_host._build_dv2_decision_plan
    patch_loaders(monkeypatch, mod, prices, universe, rec["T"])
    plan = builder(release(fam), datetime(rec["T"].year, rec["T"].month, rec["T"].day, 20), pod_state(rec))
    assert plan.decision_book_type_str == "incremental_entry_exit_book"
    assert plan.entry_target_weight_map_dict and all(abs(w - 1.0 / 3.0) <= 1e-15 for w in plan.entry_target_weight_map_dict.values())
    assert plan.entry_priority_list == [o[0] for o in rec["orders"] if not o[2]]
    assert plan.exit_asset_set == {o[0] for o in rec["orders"] if o[2]}


# --------------------------------------------------------------------------- B4 guards
def _held_state(symbol):
    return PodState(pod_id_str="pod_audit", user_id_str="audit", account_route_str="DU1",
                    position_amount_map={symbol: 100.0}, cash_float=90_000.0, total_value_float=100_000.0,
                    strategy_state_dict={"trade_id_int": 1, "current_trade_map": {symbol: 1}},
                    updated_timestamp_ts=datetime(2020, 1, 1))


def test_documents_risk_dv2_host_raises_when_held_symbol_has_no_price_column(monkeypatch):
    prices, universe = synthetic_panel()
    T = prices.index[-1]
    patch_loaders(monkeypatch, dv2_mod, prices, universe, T)
    with pytest.raises(KeyError):
        strategy_host._run_dv2_strategy_for_live_decision(release("dv2"), datetime(T.year, T.month, T.day, 20),
                                                          _held_state("GONE"))


def test_documents_risk_strategy_never_exits_a_held_name_without_a_bar_on_T():
    prices, universe = synthetic_panel()
    s = qpi_mod.QPIIbsRsiExitStrategy(name="t", benchmarks=["$SPX"], capital_base=100_000.0, **QPI_TEST_PARAMS)
    s.universe_df = universe
    sig = s.compute_signals(prices)
    row = sig.iloc[-1].copy()
    row.loc[("AAA", slice(None))] = np.nan  # held name has no bar on T (delisted)
    s._position_amount_map = {"AAA": 10.0}
    s.previous_bar = prices.index[-1]
    s.iterate(sig, row, pd.Series(dtype=float))
    assert "AAA" not in [o.asset for o in s.get_orders() if o.target]
