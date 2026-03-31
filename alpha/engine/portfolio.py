"""
portfolio.py
------------
Multi-strategy pod aggregator. Combines completed strategy runs into a unified
portfolio view using independently compounded pod equity curves.

Each strategy (pod) runs independently through run_daily() with its own capital.
The Portfolio class is a read-only aggregator that reconstructs a combined equity
curve from weighted pod returns over their common date range.
"""

import pickle
import pandas as pd
import numpy as np

from alpha.engine.plot import plot as engine_plot
from alpha.engine.metrics import (
    generate_overall_metrics,
    generate_trades_metrics,
    generate_monthly_returns,
    cross_correlation_matrix,
    diversification_ratio,
    rolling_diversification_ratio,
    rolling_pairwise_correlation,
)


class Portfolio:
    _VALID_REBALANCE = {None, 'monthly', 'quarterly', 'annually'}
    _DIAGNOSTIC_WINDOW_INT = 63

    def __init__(self, strategies: list, weights: list[float] = None,
                 name: str = 'Portfolio', capital_base: float = None,
                 rebalance: str = None, pod_info_list: list[dict] | None = None):
        """
        parameters:
        - strategies: list of Strategy objects that have been run through run_daily().
        - weights: capital allocation weights per pod. Must sum to ~1.0.
                   Defaults to equal weight (1/N).
        - name: display name for the portfolio.
        - capital_base: total portfolio capital. Defaults to sum of pod capital bases.
        - rebalance: periodic rebalancing frequency. None (default) = buy-and-hold,
                     'monthly', 'quarterly', or 'annually'.
        """
        # validate that every strategy has results
        for s in strategies:
            if s.results is None or len(s.results) == 0:
                raise ValueError(f"Strategy '{s.name}' has no results. Run it through run_daily() first.")

        n = len(strategies)

        if weights is None:
            weights = [1.0 / n] * n

        if len(weights) != n:
            raise ValueError(f"len(weights)={len(weights)} != len(strategies)={n}")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.6f}")

        if rebalance not in self._VALID_REBALANCE:
            raise ValueError(f"rebalance must be one of {self._VALID_REBALANCE}, got '{rebalance}'")

        self.strategies = strategies
        self.weights = weights
        self.name = name
        self._capital_base = capital_base if capital_base is not None else sum(
            s._capital_base for s in strategies
        )
        self._rebalance = rebalance
        self.source_config_path = None
        self.pod_info_list = self._build_pod_info_list(pod_info_list)

        # populated by _build()
        self.results = None
        self.summary = None
        self.sleeve_summary = None
        self.standalone_pod_summary = None
        self.summary_trades = None
        self.monthly_returns = None
        self._trades = None
        self._transactions = None
        self._common_start = None
        self._common_end = None
        self._daily_rets = None
        self._aligned_pod_total_value_df = None
        self._sleeve_portfolio_value_df = None
        self._pod_equities = None
        self._rebalance_date_index = pd.DatetimeIndex([])
        self.drift_weight_df = None
        self.correlation_matrix = None
        self.diversification_ratio = None
        self.target_diversification_ratio = None
        self.realized_diversification_ratio = None
        self.average_rolling_diversification_ratio = None
        self.rolling_pairwise_correlation_df = None
        self.rolling_diversification_ratio_ser = None

        self._build()

    def _build_pod_info_list(self, pod_info_list: list[dict] | None) -> list[dict]:
        if pod_info_list is not None and len(pod_info_list) != len(self.strategies):
            raise ValueError(
                f"len(pod_info_list)={len(pod_info_list)} != len(strategies)={len(self.strategies)}"
            )

        if pod_info_list is None:
            pod_info_list = [{} for _ in self.strategies]

        normalized_pod_info_list = []
        for idx, strategy in enumerate(self.strategies):
            pod_info_dict = dict(pod_info_list[idx])
            pod_info_dict.setdefault('strategy_name', strategy.name)
            pod_info_dict['weight'] = float(self.weights[idx])
            pod_info_dict['allocated_capital'] = float(self._capital_base * self.weights[idx])
            normalized_pod_info_list.append(pod_info_dict)

        return normalized_pod_info_list

    def _build(self):
        """Compute combined equity curve using buy-and-hold pod compounding.

        Each pod compounds independently from its initial capital allocation.
        This matches IBKR-realistic behavior where each pod gets a capital
        allocation and runs independently (weights drift with performance).
        Optional periodic rebalancing redistributes capital back to target weights.
        """
        # find common date range
        common_idx = self.strategies[0].results.index
        for s in self.strategies[1:]:
            common_idx = common_idx.intersection(s.results.index)
        common_idx = common_idx.sort_values()

        if len(common_idx) == 0:
            raise ValueError("No overlapping dates across strategies.")

        self._common_start = common_idx[0]
        self._common_end = common_idx[-1]

        aligned_pod_total_value_df = pd.DataFrame(index=common_idx)
        pod_invested_fraction_df = pd.DataFrame(index=common_idx)
        aligned_daily_return_df = pd.DataFrame(index=common_idx)
        for strategy_obj in self.strategies:
            pod_total_value_ser = strategy_obj.results.loc[common_idx, 'total_value'].astype(float)
            if not np.isfinite(pod_total_value_ser.iloc[0]) or float(pod_total_value_ser.iloc[0]) <= 0.0:
                raise ValueError(
                    f"Strategy '{strategy_obj.name}' has an invalid overlap start value "
                    f"{pod_total_value_ser.iloc[0]!r} on {common_idx[0]}."
                )

            aligned_pod_total_value_df[strategy_obj.name] = pod_total_value_ser

            # *** CRITICAL*** Recompute pod returns from the overlap-aligned
            # sleeve path so the first common row is a clean capital anchor.
            aligned_daily_return_df[strategy_obj.name] = (
                pod_total_value_ser.pct_change(fill_method=None).fillna(0.0)
            )

            if 'portfolio_value' in strategy_obj.results.columns:
                standalone_portfolio_value_ser = strategy_obj.results.loc[
                    common_idx, 'portfolio_value'
                ].astype(float)
                pod_invested_fraction_df[strategy_obj.name] = (
                    standalone_portfolio_value_ser
                    .div(pod_total_value_ser)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
            else:
                pod_invested_fraction_df[strategy_obj.name] = 0.0

        self._aligned_pod_total_value_df = aligned_pod_total_value_df
        self._daily_rets = aligned_daily_return_df

        # build pod equity curves - each pod compounds independently
        pod_equity_df = pd.DataFrame(index=common_idx)
        for strategy_idx_int, strategy_obj in enumerate(self.strategies):
            allocated_capital_float = float(self._capital_base * self.weights[strategy_idx_int])
            pod_return_ser = aligned_daily_return_df[strategy_obj.name].astype(float)
            pod_equity_df[strategy_obj.name] = allocated_capital_float * (1.0 + pod_return_ser).cumprod()

        # optional periodic rebalancing
        if self._rebalance is not None:
            pod_equity_df = self._apply_rebalancing(pod_equity_df, aligned_daily_return_df)

        self._pod_equities = pod_equity_df
        self._sleeve_portfolio_value_df = pod_equity_df * pod_invested_fraction_df

        # portfolio equity = sum of pod equities
        portfolio_equity_ser = pod_equity_df.sum(axis=1)
        portfolio_daily_return_ser = portfolio_equity_ser.pct_change(fill_method=None).fillna(0.0)
        self.drift_weight_df = pod_equity_df.div(portfolio_equity_ser, axis=0)

        # drawdown
        running_max_ser = portfolio_equity_ser.cummax()
        drawdown_ser = portfolio_equity_ser / running_max_ser - 1.0
        max_drawdown_ser = drawdown_ser.cummin()

        self.results = pd.DataFrame({
            'total_value': portfolio_equity_ser,
            'daily_returns': portfolio_daily_return_ser,
            'drawdown': drawdown_ser,
            'max_drawdown': max_drawdown_ser,
        }, index=common_idx)

        self._summarize()

    def _apply_rebalancing(self, pod_equities: pd.DataFrame,
                           daily_rets: pd.DataFrame) -> pd.DataFrame:
        """Redistribute pod capital back to target weights at rebalance dates.

        At each rebalance date, the total portfolio value is redistributed
        across pods at target weights, then each pod compounds forward
        independently until the next rebalance date.
        """
        freq_map = {'monthly': 'MS', 'quarterly': 'QS', 'annually': 'YS'}
        freq = freq_map[self._rebalance]
        idx = pod_equities.index

        # generate rebalance dates and snap to actual trading days
        rebal_dates = pd.date_range(start=idx[0], end=idx[-1], freq=freq)
        # *** CRITICAL*** Snap each calendar rebalance marker to the first
        # actual trading day at or after that date to avoid calendar leakage.
        positions = np.searchsorted(idx, rebal_dates, side='left')
        positions = positions[positions < len(idx)]
        rebal_trading_days = idx[np.unique(positions)]

        # skip the very first day (initial allocation is already at target)
        rebal_trading_days = rebal_trading_days[rebal_trading_days > idx[0]]
        self._rebalance_date_index = pd.DatetimeIndex(rebal_trading_days)

        weights_arr = np.array(self.weights)
        result = pod_equities.copy()

        for rebal_date in rebal_trading_days:
            # find position of rebalance date
            pos = idx.get_loc(rebal_date)
            if pos == 0:
                continue

            # total portfolio value at end of previous day
            prev_date = idx[pos - 1]
            total_val = result.loc[prev_date].sum()

            # redistribute at target weights and compound forward
            for i, s in enumerate(self.strategies):
                new_pod_capital = total_val * weights_arr[i]
                # compound from rebal_date onward using daily returns
                future_rets = daily_rets.loc[idx[pos:], s.name]
                result.loc[idx[pos:], s.name] = new_pod_capital * (1 + future_rets).cumprod()

        return result

    def _summarize(self):
        """Build summary tables, trades, monthly returns."""
        common_idx = self.results.index

        # --- overall metrics: portfolio column + weighted sleeves ---
        self.summary = pd.DataFrame()
        self.summary[self.name] = generate_overall_metrics(
            self.results['total_value'],
            capital_base=self._capital_base,
        )
        self.sleeve_summary = pd.DataFrame()
        self.standalone_pod_summary = pd.DataFrame()

        for strategy_idx_int, strategy_obj in enumerate(self.strategies):
            pct_label_str = f"{self.weights[strategy_idx_int]:.0%}"
            sleeve_col_name_str = f"{strategy_obj.name} Sleeve ({pct_label_str})"
            standalone_col_name_str = f"{strategy_obj.name} Standalone"
            allocated_capital_float = float(self._capital_base * self.weights[strategy_idx_int])

            sleeve_total_value_ser = self._pod_equities[strategy_obj.name].astype(float)
            sleeve_portfolio_value_ser = self._sleeve_portfolio_value_df[strategy_obj.name].astype(float)
            sleeve_trade_df = (
                strategy_obj._trades
                if strategy_obj._trades is not None and len(strategy_obj._trades) > 0
                else None
            )
            self.summary[sleeve_col_name_str] = generate_overall_metrics(
                sleeve_total_value_ser,
                trades=sleeve_trade_df,
                portfolio_value=sleeve_portfolio_value_ser,
                capital_base=allocated_capital_float,
            )
            self.sleeve_summary[sleeve_col_name_str] = self.summary[sleeve_col_name_str]

            standalone_total_value_ser = strategy_obj.results.loc[common_idx, 'total_value'].astype(float)
            standalone_trade_df = (
                strategy_obj._trades
                if strategy_obj._trades is not None and len(strategy_obj._trades) > 0
                else None
            )
            standalone_portfolio_value_ser = (
                strategy_obj.results.loc[common_idx, 'portfolio_value'].astype(float)
                if 'portfolio_value' in strategy_obj.results.columns else None
            )
            standalone_commission_float = None
            transaction_df = strategy_obj.get_transactions()
            if transaction_df is not None and len(transaction_df) > 0 and 'commission' in transaction_df.columns:
                standalone_commission_float = float(transaction_df['commission'].sum())

            self.standalone_pod_summary[standalone_col_name_str] = generate_overall_metrics(
                standalone_total_value_ser,
                trades=standalone_trade_df,
                portfolio_value=standalone_portfolio_value_ser,
                capital_base=strategy_obj._capital_base,
                total_commissions=standalone_commission_float,
            )

        # --- concatenate trades with pod column ---
        all_trades = []
        for strategy_obj in self.strategies:
            if strategy_obj._trades is not None and len(strategy_obj._trades) > 0:
                t = strategy_obj._trades.copy()
                t['pod'] = strategy_obj.name
                all_trades.append(t)
        self._trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

        # --- concatenate transactions with pod column ---
        all_txns = []
        for strategy_obj in self.strategies:
            txns = strategy_obj.get_transactions()
            if txns is not None and len(txns) > 0:
                t = txns.copy()
                t['pod'] = strategy_obj.name
                all_txns.append(t)
        self._transactions = pd.concat(all_txns, ignore_index=True) if all_txns else pd.DataFrame()

        # --- trade metrics on concatenated trades ---
        if len(self._trades) > 0:
            self.summary_trades = generate_trades_metrics(self._trades, self.results.index)
        else:
            self.summary_trades = pd.DataFrame()

        # --- monthly returns on combined equity ---
        self.monthly_returns = generate_monthly_returns(
            self.results['total_value'].copy(),
            add_sharpe_ratios=True,
            add_max_drawdowns=True,
        )

        # --- total commissions ---
        if len(self._transactions) > 0 and 'commission' in self._transactions.columns:
            self.total_commissions = self._transactions['commission'].sum()
        else:
            self.total_commissions = 0.0

        # --- cross-strategy diagnostics ---
        realized_daily_return_df = self._daily_rets.iloc[1:].copy()
        realized_weight_df = self.drift_weight_df.loc[realized_daily_return_df.index].copy()
        self.correlation_matrix = cross_correlation_matrix(realized_daily_return_df)
        self.target_diversification_ratio = diversification_ratio(realized_daily_return_df, self.weights)
        self.diversification_ratio = self.target_diversification_ratio

        if len(realized_daily_return_df) > 0:
            self.realized_diversification_ratio = diversification_ratio(
                realized_daily_return_df,
                realized_weight_df.iloc[-1].to_list(),
            )
            self.rolling_pairwise_correlation_df = rolling_pairwise_correlation(
                realized_daily_return_df,
                window_int=self._DIAGNOSTIC_WINDOW_INT,
            )
            self.rolling_diversification_ratio_ser = rolling_diversification_ratio(
                realized_daily_return_df,
                realized_weight_df,
                window_int=self._DIAGNOSTIC_WINDOW_INT,
            )
            if len(self.rolling_diversification_ratio_ser.dropna()) > 0:
                self.average_rolling_diversification_ratio = float(
                    self.rolling_diversification_ratio_ser.dropna().mean()
                )
            else:
                self.average_rolling_diversification_ratio = np.nan
        else:
            self.realized_diversification_ratio = np.nan
            self.rolling_pairwise_correlation_df = pd.DataFrame(index=self.results.index)
            self.rolling_diversification_ratio_ser = pd.Series(
                dtype=float,
                name='rolling_diversification_ratio_ser',
            )
            self.average_rolling_diversification_ratio = np.nan

    def plot(self, save_to=None):
        """Plot combined equity curve with per-pod overlays and benchmark."""
        common_idx = self.results.index

        # build additional_returns: one column per pod, normalized to start at 1.0
        additional = self._pod_equities.loc[common_idx].copy()

        # find a benchmark from the first strategy that has one
        benchmark_tv = None
        benchmark_label = 'Benchmark'
        for strategy_obj in self.strategies:
            if hasattr(strategy_obj, '_benchmarks') and len(strategy_obj._benchmarks) > 0:
                bm = strategy_obj._benchmarks[0]
                if bm in strategy_obj.results.columns:
                    benchmark_tv = strategy_obj.results.loc[common_idx, bm].astype(float)
                    benchmark_label = bm
                    break

        engine_plot(
            strategy_total_value=self.results['total_value'],
            strategy_drawdown=self.results['drawdown'],
            benchmark_total_value=benchmark_tv,
            benchmark_label=benchmark_label,
            strategy_label=self.name,
            additional_returns=additional,
            alpha_additional=0.25,
            save_to=save_to,
        )

    def to_pickle(self, path):
        """Save the portfolio object to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read_pickle(path):
        """Load a portfolio object from a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        pods = ', '.join(f"{s.name}({w:.0%})" for s, w in zip(self.strategies, self.weights))
        return f"Portfolio('{self.name}', pods=[{pods}], capital={self._capital_base:,.0f})"


