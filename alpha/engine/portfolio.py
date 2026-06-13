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
    generate_tail_risk_diagnostics,
    rolling_diversification_ratio,
    rolling_pairwise_correlation,
)


class Portfolio:
    _VALID_REBALANCE = {None, 'monthly', 'quarterly', 'annually'}
    _VALID_REBALANCE_POLICY = {'fixed', 'equal', 'inverse_volatility'}
    _DEFAULT_INVERSE_VOLATILITY_LOOKBACK_DAY_INT = 60
    _DIAGNOSTIC_WINDOW_INT = 63
    _TAIL_FRACTION_FLOAT = 0.05
    _MIN_TAIL_DAYS_INT = 1

    def __init__(self, strategies: list, weights: list[float] = None,
                 name: str = 'Portfolio', capital_base: float = None,
                 rebalance: str = None, pod_info_list: list[dict] | None = None,
                 rebalance_policy_str: str = 'fixed',
                 rebalance_inverse_volatility_lookback_day_int: int = None):
        """
        parameters:
        - strategies: list of Strategy objects that have been run through run_daily().
        - weights: capital allocation weights per pod. Must sum to ~1.0.
                   Defaults to equal weight (1/N).
        - name: display name for the portfolio.
        - capital_base: total portfolio capital. Defaults to sum of pod capital bases.
        - rebalance: periodic rebalancing frequency. None (default) = buy-and-hold,
                     'monthly', 'quarterly', or 'annually'.
        - rebalance_policy_str: target-weight policy used on rebalance dates.
                                'fixed' returns to configured weights, 'equal'
                                returns to 1/N, and 'inverse_volatility' uses
                                trailing pod-return volatility.
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

        rebalance_policy_str = str(rebalance_policy_str).strip().lower()
        if rebalance_policy_str not in self._VALID_REBALANCE_POLICY:
            raise ValueError(
                f"rebalance_policy_str must be one of {self._VALID_REBALANCE_POLICY}, "
                f"got '{rebalance_policy_str}'"
            )

        if rebalance_inverse_volatility_lookback_day_int is None:
            rebalance_inverse_volatility_lookback_day_int = (
                self._DEFAULT_INVERSE_VOLATILITY_LOOKBACK_DAY_INT
            )
        rebalance_inverse_volatility_lookback_day_int = int(
            rebalance_inverse_volatility_lookback_day_int
        )
        if rebalance_inverse_volatility_lookback_day_int <= 1:
            raise ValueError("rebalance_inverse_volatility_lookback_day_int must be greater than 1.")

        self.strategies = strategies
        self.weights = weights
        self.name = name
        self._capital_base = capital_base if capital_base is not None else sum(
            s._capital_base for s in strategies
        )
        self._rebalance = rebalance
        self._rebalance_policy = rebalance_policy_str
        self._rebalance_inverse_volatility_lookback_day_int = (
            rebalance_inverse_volatility_lookback_day_int
        )
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
        self._scheduled_rebalance_date_index = pd.DatetimeIndex([])
        self.rebalance_target_weight_df = pd.DataFrame()
        self.rebalance_diagnostic_df = pd.DataFrame()
        self.drift_weight_df = None
        self.correlation_matrix = None
        self.diversification_ratio = None
        self.target_diversification_ratio = None
        self.realized_diversification_ratio = None
        self.average_rolling_diversification_ratio = None
        self.rolling_pairwise_correlation_df = None
        self.rolling_diversification_ratio_ser = None
        self.tail_event_date_index = pd.DatetimeIndex([])
        self.tail_return_df = pd.DataFrame()
        self.tail_correlation_matrix = pd.DataFrame()
        self.tail_contribution_df = pd.DataFrame()
        self.tail_summary_df = pd.DataFrame()

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

    def _rebalance_target_weight_ser(
        self,
        rebalance_date,
        rebalance_position_int: int,
        daily_return_df: pd.DataFrame,
    ) -> tuple[pd.Series | None, dict]:
        strategy_name_list = [strategy_obj.name for strategy_obj in self.strategies]
        diagnostic_dict = {
            'rebalance_date': pd.Timestamp(rebalance_date),
            'policy_str': self._rebalance_policy,
            'lookback_day_int': (
                self._rebalance_inverse_volatility_lookback_day_int
                if self._rebalance_policy == 'inverse_volatility'
                else None
            ),
            'observation_count_int': None,
            'status_str': 'applied',
        }

        if self._rebalance_policy == 'fixed':
            target_weight_ser = pd.Series(
                np.array(self.weights, dtype=float),
                index=strategy_name_list,
                dtype=float,
            )
            return target_weight_ser, diagnostic_dict

        if self._rebalance_policy == 'equal':
            target_weight_ser = pd.Series(
                1.0 / float(len(strategy_name_list)),
                index=strategy_name_list,
                dtype=float,
            )
            return target_weight_ser, diagnostic_dict

        lookback_day_int = int(self._rebalance_inverse_volatility_lookback_day_int)

        # *** CRITICAL*** Inverse-volatility PM targets are decided before the
        # rebalance-date return is known. The window below excludes the current
        # rebalance row and also excludes the overlap anchor row whose return is
        # forced to 0.0 by construction.
        available_return_df = daily_return_df.iloc[1:rebalance_position_int]
        window_return_df = available_return_df.tail(lookback_day_int)
        diagnostic_dict['observation_count_int'] = int(len(window_return_df))
        if len(window_return_df) < lookback_day_int:
            diagnostic_dict['status_str'] = 'skipped_insufficient_history'
            return None, diagnostic_dict

        volatility_ser = window_return_df.std(ddof=0) * float(np.sqrt(252.0))
        invalid_volatility_name_list = [
            strategy_name_str
            for strategy_name_str, volatility_float in volatility_ser.items()
            if not np.isfinite(float(volatility_float)) or float(volatility_float) <= 0.0
        ]
        if len(invalid_volatility_name_list) > 0:
            raise ValueError(
                "Inverse-volatility rebalance has invalid trailing volatility on "
                f"{pd.Timestamp(rebalance_date).date()}: "
                + ", ".join(invalid_volatility_name_list)
            )

        inverse_volatility_ser = 1.0 / volatility_ser
        inverse_volatility_sum_float = float(inverse_volatility_ser.sum())
        if not np.isfinite(inverse_volatility_sum_float) or inverse_volatility_sum_float <= 0.0:
            raise ValueError(
                "Inverse-volatility rebalance has invalid inverse-volatility sum on "
                f"{pd.Timestamp(rebalance_date).date()}."
            )

        target_weight_ser = inverse_volatility_ser / inverse_volatility_sum_float
        return target_weight_ser.astype(float), diagnostic_dict

    def _apply_rebalancing(self, pod_equities: pd.DataFrame,
                           daily_rets: pd.DataFrame) -> pd.DataFrame:
        """Redistribute pod capital back to target weights at rebalance dates.

        At each rebalance date, the previous close-marked total portfolio value
        is redistributed across pods at target weights, then each pod compounds
        forward independently until the next rebalance date.

        Formulas for inverse-volatility policy:

            r_{i,t} = E_{i,t} / E_{i,t-1} - 1
            sigma_{i,d} = std(r_i over the trailing L completed days before d) * sqrt(252)
            w_{i,d} = (1 / sigma_{i,d}) / sum_j(1 / sigma_{j,d})
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
        rebal_trading_days = pd.DatetimeIndex(rebal_trading_days[rebal_trading_days > idx[0]])
        self._scheduled_rebalance_date_index = rebal_trading_days

        result = pod_equities.copy()
        target_weight_ser_list = []
        diagnostic_dict_list = []
        applied_rebalance_date_list = []

        for rebal_date in rebal_trading_days:
            # find position of rebalance date
            pos = idx.get_loc(rebal_date)
            if pos == 0:
                continue

            target_weight_ser, diagnostic_dict = self._rebalance_target_weight_ser(
                rebalance_date=rebal_date,
                rebalance_position_int=int(pos),
                daily_return_df=daily_rets,
            )
            diagnostic_dict_list.append(diagnostic_dict)
            if target_weight_ser is None:
                continue

            target_weight_sum_float = float(target_weight_ser.sum())
            if abs(target_weight_sum_float - 1.0) > 1e-9:
                raise ValueError(
                    "Rebalance target weights must sum to 1.0 on "
                    f"{pd.Timestamp(rebal_date).date()}, got {target_weight_sum_float:.12f}."
                )

            applied_rebalance_date_list.append(rebal_date)
            target_weight_ser.name = pd.Timestamp(rebal_date)
            target_weight_ser_list.append(target_weight_ser.copy())

            # total portfolio value at end of previous day
            prev_date = idx[pos - 1]
            total_val = result.loc[prev_date].sum()

            # redistribute at target weights and compound forward
            for strategy_obj in self.strategies:
                new_pod_capital = total_val * float(target_weight_ser.loc[strategy_obj.name])
                # compound from rebal_date onward using daily returns
                future_rets = daily_rets.loc[idx[pos:], strategy_obj.name]
                result.loc[idx[pos:], strategy_obj.name] = new_pod_capital * (1 + future_rets).cumprod()

        self._rebalance_date_index = pd.DatetimeIndex(applied_rebalance_date_list)
        if len(target_weight_ser_list) > 0:
            self.rebalance_target_weight_df = pd.DataFrame(target_weight_ser_list)
            self.rebalance_target_weight_df.index.name = 'rebalance_date'
        else:
            self.rebalance_target_weight_df = pd.DataFrame(columns=pod_equities.columns, dtype=float)
            self.rebalance_target_weight_df.index.name = 'rebalance_date'

        if len(diagnostic_dict_list) > 0:
            self.rebalance_diagnostic_df = pd.DataFrame(diagnostic_dict_list).set_index('rebalance_date')
            self.rebalance_diagnostic_df.index = pd.to_datetime(self.rebalance_diagnostic_df.index)
        else:
            self.rebalance_diagnostic_df = pd.DataFrame(
                columns=['policy_str', 'lookback_day_int', 'observation_count_int', 'status_str']
            )
            self.rebalance_diagnostic_df.index.name = 'rebalance_date'

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
        target_diversification_weight_list = list(self.weights)
        if self.rebalance_target_weight_df is not None and len(self.rebalance_target_weight_df) > 0:
            target_diversification_weight_list = (
                self.rebalance_target_weight_df
                .iloc[-1]
                .reindex(realized_daily_return_df.columns)
                .fillna(0.0)
                .astype(float)
                .to_list()
            )
        self.target_diversification_ratio = diversification_ratio(
            realized_daily_return_df,
            target_diversification_weight_list,
        )
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

        tail_diagnostic_dict = generate_tail_risk_diagnostics(
            pod_daily_return_df=self._daily_rets,
            portfolio_daily_return_ser=self.results['daily_returns'],
            pod_equity_df=self._pod_equities,
            tail_fraction_float=self._TAIL_FRACTION_FLOAT,
            min_tail_days_int=self._MIN_TAIL_DAYS_INT,
        )
        self.tail_event_date_index = tail_diagnostic_dict['tail_event_date_index']
        self.tail_return_df = tail_diagnostic_dict['tail_return_df']
        self.tail_correlation_matrix = tail_diagnostic_dict['tail_correlation_matrix']
        self.tail_contribution_df = tail_diagnostic_dict['tail_contribution_df']
        self.tail_summary_df = tail_diagnostic_dict['tail_summary_df']

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


