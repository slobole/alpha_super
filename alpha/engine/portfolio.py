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
    generate_trades,
    generate_trades_metrics,
    generate_monthly_returns,
    cross_correlation_matrix,
    diversification_ratio,
)


class Portfolio:
    _VALID_REBALANCE = {None, 'monthly', 'quarterly', 'annually'}

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
        self.summary_trades = None
        self.monthly_returns = None
        self._trades = None
        self._transactions = None
        self._common_start = None
        self._common_end = None
        self._daily_rets = None
        self._pod_equities = None
        self.correlation_matrix = None
        self.diversification_ratio = None

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

        # extract daily returns for each pod, aligned to common dates
        daily_rets = pd.DataFrame(index=common_idx)
        for s in self.strategies:
            rets = s.results.loc[common_idx, 'daily_returns'].astype(float)
            daily_rets[s.name] = rets
        self._daily_rets = daily_rets

        # build pod equity curves — each pod compounds independently
        pod_equities = pd.DataFrame(index=common_idx)
        for i, s in enumerate(self.strategies):
            pod_capital = self._capital_base * self.weights[i]
            rets = daily_rets[s.name]
            pod_equities[s.name] = pod_capital * (1 + rets).cumprod()

        # optional periodic rebalancing
        if self._rebalance is not None:
            pod_equities = self._apply_rebalancing(pod_equities, daily_rets)

        self._pod_equities = pod_equities

        # portfolio equity = sum of pod equities
        portfolio_equity = pod_equities.sum(axis=1)
        portfolio_daily_returns = portfolio_equity.pct_change(fill_method=None).fillna(0)

        # drawdown
        running_max = portfolio_equity.cummax()
        drawdown = portfolio_equity / running_max - 1
        max_drawdown = drawdown.cummin()

        self.results = pd.DataFrame({
            'total_value': portfolio_equity,
            'daily_returns': portfolio_daily_returns,
            'drawdown': drawdown,
            'max_drawdown': max_drawdown,
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
        # snap each to the nearest trading day at or after
        positions = np.searchsorted(idx, rebal_dates, side='left')
        positions = positions[positions < len(idx)]
        rebal_trading_days = idx[np.unique(positions)]

        # skip the very first day (initial allocation is already at target)
        rebal_trading_days = rebal_trading_days[rebal_trading_days > idx[0]]

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

        # --- overall metrics: portfolio column ---
        self.summary = pd.DataFrame()
        self.summary[self.name] = generate_overall_metrics(
            self.results['total_value'],
            capital_base=self._capital_base,
        )

        # --- per-pod columns (sliced to common dates) ---
        for i, s in enumerate(self.strategies):
            pct_label = f"{self.weights[i]:.0%}"
            col_name = f"{s.name} ({pct_label})"
            pod_tv = s.results.loc[common_idx, 'total_value'].astype(float)
            pod_trades = s._trades if s._trades is not None else None
            pod_pv = s.results.loc[common_idx, 'portfolio_value'].astype(float) if 'portfolio_value' in s.results.columns else None
            total_commissions = s.get_transactions()['commission'].sum() if hasattr(s, '_transactions') else None
            self.summary[col_name] = generate_overall_metrics(
                pod_tv,
                trades=pod_trades,
                portfolio_value=pod_pv,
                capital_base=s._capital_base,
                total_commissions=total_commissions,
            )

        # --- concatenate trades with pod column ---
        all_trades = []
        for s in self.strategies:
            if s._trades is not None and len(s._trades) > 0:
                t = s._trades.copy()
                t['pod'] = s.name
                all_trades.append(t)
        self._trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

        # --- concatenate transactions with pod column ---
        all_txns = []
        for s in self.strategies:
            txns = s.get_transactions()
            if txns is not None and len(txns) > 0:
                t = txns.copy()
                t['pod'] = s.name
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
        self.correlation_matrix = cross_correlation_matrix(self._daily_rets)
        self.diversification_ratio = diversification_ratio(self._daily_rets, self.weights)

    def plot(self, save_to=None):
        """Plot combined equity curve with per-pod overlays and benchmark."""
        common_idx = self.results.index

        # build additional_returns: one column per pod, normalized to start at 1.0
        additional = pd.DataFrame(index=common_idx)
        for s in self.strategies:
            pod_tv = s.results.loc[common_idx, 'total_value'].astype(float)
            additional[s.name] = pod_tv

        # find a benchmark from the first strategy that has one
        benchmark_tv = None
        benchmark_label = 'Benchmark'
        for s in self.strategies:
            if hasattr(s, '_benchmarks') and len(s._benchmarks) > 0:
                bm = s._benchmarks[0]
                if bm in s.results.columns:
                    benchmark_tv = s.results.loc[common_idx, bm].astype(float)
                    benchmark_label = bm
                    break

        engine_plot(
            strategy_total_value=self.results['total_value'],
            strategy_drawdown=self.results['drawdown'],
            benchmark_total_value=benchmark_tv,
            benchmark_label=benchmark_label,
            strategy_label=self.name,
            additional_returns=additional,
            alpha_additional=0.5,
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
