# alpha/engine/strategy.py

"""
strategy.py
------------
abstract base class for all trading strategies used in the alpha backtesting engine.

this class defines the standard interface that any strategy must implement
to be compatible with the engine, including methods for signal computation,
order placement, performance tracking, and result summarization.

subclasses must override compute_signals().

-----------
life cycle of an order:
1. iterate() places the order Ã¢â€ â€™ stored in self._orders
2. process_orders() executes orders
3. Executed orders are recorded via add_transaction()
4. Positions updated by summing historical transactions (get_position())

"""

import pandas as pd
import numpy as np
import pickle

from abc import ABC, abstractmethod
from alpha.engine.plot import plot
from alpha.engine.order import Order, MarketOrder, LimitOrder, StopOrder
from alpha.engine.metrics import generate_drawdowns, generate_monthly_returns, generate_overall_metrics, generate_trades, generate_trades_metrics, sharpe_ratio

class Strategy(ABC):
    def __init__(self, name: str, benchmarks: list | tuple, capital_base = 10_000, slippage: float = 0.0001,
                 commission_per_share: float = 0.005, commission_minimum: float = 1.0):
        # strategy metadata
        self.name = name  # name of the strategy for identification

        # capital and portfolio management
        self._capital_base = capital_base  # initial capital allocated for the strategy
        self.cash = self._capital_base  # available cash balance in portfolio
        self.portfolio_value = 0  # current value of all open positions (EXCLUDING cash)
        self.total_value = capital_base  # total portfolio value (cash + positions)
        self._slippage = slippage  # slippage percentage applied to orders
        self._commission_per_share = commission_per_share  # IBKR default: $0.005/share
        self._commission_minimum = commission_minimum  # IBKR default: $1.00 minimum per order
        self._benchmarks = benchmarks  # list of benchmark assets for performance comparison

        # data storage and results tracking
        self.results = self.initialize_results()  # DataFrame to store performance metrics throughout the simulation
        self.additional_metrics = pd.DataFrame()  # placeholder for extra user-defined metrics

        # order and transaction tracking
        self._orders: list[Order] = []  # list of active/pending orders
        self._transactions = self.initialize_transactions()  # pd.DateFrame - A record of executed trades, including details like price, volume, and timestamps

        # trade and performance statistics
        self._trades = None  # placeholder for trade history (filled orders)
        self._drawdowns = None  # placeholder for drawdown analysis results
        self.summary = None  # summary of overall strategy performance
        self.summary_trades = None  # summary of trade-level performance metrics

        # simulation state tracking
        self.current_bar = None  # the current timestamp in the simulation (i.e., the active trading day)
        self.previous_bar = None  # the timestamp of the previous trading day, used for reference
        self.signal_audit_sample_size = getattr(self, 'signal_audit_sample_size', 10)
        self.enable_signal_audit = getattr(self, 'enable_signal_audit', False)


    # ------------------------------------------ 0 - INITIALIZATION  ------------------------------------------ #
    # initialization methods
    @staticmethod
    def initialize_transactions() -> pd.DataFrame:
        return pd.DataFrame(columns=['trade_id', 'bar', 'asset', 'amount', 'price', 'total_value', 'order_id', 'commission'])

    def _compute_commission(self, shares):
        """IBKR-style: max(minimum, per_share * |shares|). Override for custom models."""
        if self._commission_per_share == 0:
            return 0.0
        return max(self._commission_minimum, self._commission_per_share * abs(shares))
    
    def initialize_results(self) -> pd.DataFrame:
        columns = ['portfolio_value', 'cash', 'total_value', 'daily_returns',
                'total_returns', 'annualized_returns',
                'annualized_volatility', 'sharpe_ratio', 'drawdown',
                'max_drawdown']
        for benchmark in self._benchmarks:
            columns.append(benchmark)
            columns.append(f'{benchmark}_drawdown')
            columns.append(f'{benchmark}_max_drawdown')
        return pd.DataFrame(columns=columns)
    # ------------------------------------------ 1 - SIMULATION LIFECYCLE  ------------------------------------------ #
    # signal generation -> should be implemented in the subclass before the simulation initiation
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        """
        prepares and computes trading signals before the simulation starts.
        this method should be overridden in derived classes to generate
        strategy-specific signals based on historical data.

        parameters:
        - pricing_data: DataFrame containing historical market data.

        returns:
        - A modified DataFrame with computed signals.
        """
        return pricing_data

    def signal_audit_fields(self, pricing_data: pd.DataFrame, signal_data: pd.DataFrame):
        """
        Returns the columns that should be checked by the anti-lookahead audit.

        By default, only columns added by `compute_signals()` are audited.
        Strategies can override this when they want to audit a custom subset.
        """
        original_cols = set(pricing_data.columns)
        return [col for col in signal_data.columns if col not in original_cols]

    def audit_signals(self, pricing_data: pd.DataFrame, signal_data: pd.DataFrame, sample_size: int | None = None):
        """
        Recompute signals on truncated histories and compare sampled rows to the
        full-history result. This catches accidental future leakage for derived
        fields created inside `compute_signals()`.
        """
        audit_cols = self.signal_audit_fields(pricing_data, signal_data)
        if len(audit_cols) == 0:
            return

        if not signal_data.index.equals(pricing_data.index):
            raise ValueError(
                f"{self.name}: compute_signals() must preserve the original index order."
            )

        sample_size = self.signal_audit_sample_size if sample_size is None else sample_size
        if sample_size is None or sample_size <= 0:
            return

        sample_positions = np.linspace(
            0,
            len(pricing_data.index) - 1,
            min(sample_size, len(pricing_data.index)),
            dtype=int,
        )
        sampled_index = pricing_data.index[np.unique(sample_positions)]

        for bar in sampled_index:
            truncated_input = pricing_data.loc[:bar]
            truncated_signal_data = self.compute_signals(truncated_input.copy())

            missing_cols = [col for col in audit_cols if col not in truncated_signal_data.columns]
            if missing_cols:
                raise ValueError(
                    f"{self.name}: compute_signals() did not return audited fields {missing_cols} "
                    f"when recomputed through {bar}."
                )

            for col in audit_cols:
                full_value = signal_data.loc[bar, col]
                truncated_value = truncated_signal_data.loc[bar, col]
                if not self._signal_values_match(full_value, truncated_value):
                    raise ValueError(
                        f"{self.name}: possible lookahead leakage in feature {col} at {bar}. "
                        "The full-history value differs from the value recomputed using only data available up to that bar."
                    )

    @staticmethod
    def _signal_values_match(full_value, truncated_value, atol: float = 1e-12) -> bool:
        if pd.isna(full_value) and pd.isna(truncated_value):
            return True
        if isinstance(full_value, (float, np.floating)) or isinstance(truncated_value, (float, np.floating)):
            return bool(np.isclose(full_value, truncated_value, atol=atol, rtol=0, equal_nan=True))
        return full_value == truncated_value
    
    # trade logic -> should be implemented in the subclass
    @abstractmethod
    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        """
        defines the core trading logic for each iteration of the simulation.
        this method must be implemented in derived strategy classes.
        it determines trading decisions at the end of the period, placing
        orders to be executed at the next market open.

        parameters:
        - data: DataFrame containing available market data up to the current bar.
        - close: DataFrame containing closing prices for the previous bar.
        - open_prices: Series containing opening prices for the current bar.
        """
        pass

    # process orders
    def process_orders(self, prices: pd.DataFrame):
        """
        executes all active orders based on current market prices and ensures that orders properly executed,
        then updates the portfolio state. 

        also, it ensures that the simulation stays consistent with the real trading conditions.

        Parameters:
        - prices: A DataFrame containing OHLC prices for all assets.
        """

        # 1. loop through all active orders
        # 2. for each order, calculate the transaction amount in shares and the execution price
        # 3. record the transaction in the transactions DataFrame
        # 4. update the simulation's main state variables: cash, portfolio value, and total value to reflect the executed trades

        # ensure the current bar exist in the prices data
        if self.current_bar not in prices.index:
            return

        executed_orders = []  # list to store successfully executed orders

        # loop through all active orders
        for order in self.get_orders():
            # retrieve current market prices for the asset
            if order.asset in prices.columns.get_level_values(0):
                current_open = prices.loc[self.current_bar, (order.asset, 'Open')]
                current_high = prices.loc[self.current_bar, (order.asset, 'High')]
                current_low = prices.loc[self.current_bar, (order.asset, 'Low')]
            else:
                raise RuntimeError(f"{order.asset} not in available")


            # compute the approximate order amount based on the current position and portfolio value
            position = self.get_position(order.asset)

            # check if the asset has valid market data; if not, close the position
            if pd.isna(current_open):
                # this sometimes happens when an asset is delisted. in that case, we
                # close the position using last closing price and warn the user.
                print(f'Asset {order.asset} not found in prices on {self.current_bar}')
                price = prices[(order.asset, 'Close')].dropna().iloc[-1]
                amount = order.amount_in_shares(price, self.total_value, position)
                commission = self._compute_commission(amount)
                self.add_transaction(order.trade_id, self.current_bar, order.asset,
                                    amount, price, price * amount, order.id, commission)
                executed_orders.append(order)
                continue

            amount_approx = order.amount_in_shares(current_open, self.total_value, position)
            # apply slippage (penalty) to execution price
            penalty = 1 + np.sign(amount_approx) * self._slippage  # slippage adjustment for the amount

            # --- ORDER EXECUTION LOGIC ---
            if isinstance(order, MarketOrder):
                # market orders execute at the **opening price** of the current day
                # (liquidity/cash constraints are ignored here; this can be improved)
                price = current_open * penalty
                amount = order.amount_in_shares(price, self.total_value, position)
                commission = self._compute_commission(amount)
                self.add_transaction(order.trade_id, self.current_bar, order.asset, amount, price,
                                    price * amount, order.id, commission)
                executed_orders.append(order)

            elif isinstance(order, LimitOrder):
                # limit orders execute only if the limit price is within the day's range
                if ((amount_approx > 0 and current_low <= order.limit_price) or
                        (amount_approx < 0 and current_high >= order.limit_price)):
                    # determine execution price: best possible price without exceeding the limit
                    if amount_approx > 0:
                        price = min(order.limit_price,
                                    current_open) * penalty  # buy at the best valid price
                    else:
                        price = max(order.limit_price,
                                    current_open) * penalty  # sell at the best valid price

                    amount_exact = order.amount_in_shares(price, self.total_value, position)
                    commission = self._compute_commission(amount_exact)
                    self.add_transaction(order.trade_id, self.current_bar, order.asset,
                                        amount_exact, price, price * amount_exact, order.id, commission)
                    executed_orders.append(order)
                else:
                    # if the order is not triggered, remove it
                    self.remove_order(order)

            elif isinstance(order, StopOrder):
                # stop orders trigger when the stop price is reached
                if ((amount_approx > 0 and current_high >= order.stop_price) or
                        (amount_approx < 0 and current_low <= order.stop_price)):
                    # determine execution price: worst valid price at or beyond the stop level
                    if amount_approx > 0:
                        price = max(order.stop_price,
                                    current_open) * penalty  # buy at stop or worse
                    else:
                        price = min(order.stop_price,
                                    current_open) * penalty  # sell at stop or worse

                    amount_exact = order.amount_in_shares(price, self.total_value, position)
                    commission = self._compute_commission(amount_exact)
                    self.add_transaction(order.trade_id, self.current_bar, order.asset,
                                        amount_exact, price, price * amount_exact, order.id, commission)
                    executed_orders.append(order)

        # remove all executed orders from the active order list
        for order in executed_orders:
            self.remove_order(order)

        # --- UPDATE PORTFOLIO STATE ---

        # deduct cash used for executed transactions (trade value + commissions)
        today_txns = self.get_transactions(self.current_bar)
        self.cash -= today_txns['total_value'].sum()
        self.cash -= today_txns['commission'].sum()

        # compute the updated portfolio value
        pos = self.get_positions()
        if len(pos) == 0:
            self.portfolio_value = 0  # No positions left
        else:
            # retrieve closing prices to value the current portfolio
            close_prices = prices.loc[self.current_bar, (slice(None), 'Close')]
            close_prices.index = close_prices.index.get_level_values(0)
            self.portfolio_value = (pos * close_prices).sum()

        # update the total account value (cash + portfolio holdings)
        self.total_value = self.cash + self.portfolio_value

    # update performance metrics
    def update_metrics(self, prices: pd.DataFrame, start: pd.Timestamp):
        """
        updates the portfolio's performance metrics at the end of each trading day.

        parameters:
        - prices: DataFrame containing asset price data.
        - start: Timestamp indicating the start date of the simulation.
        """
        tdy = self.current_bar.date()  # extract the date from the current trading bar
        # if this is the first trading day (no previous bar exists), initialize metrics
        if self.previous_bar is None:
            daily_return = 0  # no return on the first day
            total_return = 0  # no total return yet
            annualized_return = 0  # cannot annualize returns with one data point
            annualized_volatility = 0  # no volatility calculated yet
            sr = 0  # sharpe ratio is undefined initially
            drawdown = 0  # no drawdown on the first day
            max_drawdown = 0  # no max drawdown on the first day
            # initialize benchmark values
            bench = []
            for _ in self._benchmarks:
                bench.append((self._capital_base, 0, 0))  # starting capital, no drawdown

        else:
            # calculate daily return based on total value change
            daily_return = self.total_value / self.previous_total_value - 1
            # compute total return since the start of the simulation
            total_return = self.total_value / self._capital_base - 1
            # compute annualized return, avoiding division by zero
            if self.num_days == 0:
                annualized_return = np.nan
            else:
                annualized_return = (1 + total_return) ** (252 / self.num_days) - 1

            # append the latest daily return and portfolio values to historical series
            drs = pd.Series(np.append(self.daily_returns_series, daily_return))
            pvs = pd.Series(np.append(self.portfolio_value_series, self.portfolio_value))
            tvs = pd.Series(np.append(self.total_value_series, self.total_value))
            # calculate annualized volatility (standard deviation of daily returns)
            annualized_volatility = drs.std() * np.sqrt(252)
            # compute the Sharpe ratio only if volatility is non-zero
            sr = sharpe_ratio(drs, pvs) if annualized_volatility != 0 else np.nan  # mean/std of daily returns
            # calculate drawdown (percentage decline from highest portfolio value)
            drawdown = self.total_value / max(tvs) - 1
            # compute max drawdown (worst observed drawdown)
            max_drawdown = (tvs / tvs.cummax() - 1).min()

            # initialize benchmark metrics
            bench = []
            for benchmark in self._benchmarks:
                # retrieve closing price of the benchmark for the current and start dates
                bc = prices.loc[self.current_bar, (benchmark, 'Close')]
                b0 = prices.loc[start, (benchmark, 'Close')]
                # compute the benchmark's total value assuming it started with the same capital
                bench_val = bc / b0 * self._capital_base
                # append latest benchmark value to historical series
                bvs = pd.Series(np.append(self.benchmark_value_series(benchmark), bench_val))
                # compute benchmark drawdown and max drawdown
                bench_drawdown = bench_val / max(bvs) - 1
                bench_max_drawdown = (bvs / bvs.cummax() - 1).min()
                # store benchmark metrics
                bench.append((bench_val, bench_drawdown, bench_max_drawdown))

        # store the computed metrics for the current day in the results DataFrame
        self.results.loc[tdy] = [
            self.portfolio_value, self.cash, self.total_value,
            daily_return, total_return, annualized_return, annualized_volatility, sr,
            drawdown, max_drawdown] + [item for tup in bench for item in tup]

    # finish computations and extras -> should be implemented in the subclass
    def finalize(self, current_data: pd.DataFrame):
        """
        performs final computations after the simulation ends.
        this method should be overridden in derived classes to handle
        post-simulation tasks such as logging, reporting, or further analysis.

        parameters:
        - current_data: DataFrame containing the final state of market data.
        """
        pass

    # summarize results
    def summarize(self, include_benchmarks=True):
        """
        generates a summary of the strategy's performance, including
        trades and drawdowns.

        parameters:
        - include_benchmarks: if True, includes benchmark performance in the summary.

        stores:
        - self.summary: A DataFrame containing key performance metrics.
        - self.summary_trades: A DataFrame summarizing trade statistics.
        """
        # generate trade history and drawdown metrics
        self._trades = generate_trades(self.get_transactions())
        self._drawdowns = generate_drawdowns(self.results['drawdown'])
        # compute strategy performance metrics
        self.summary = pd.DataFrame()
        total_commissions = self.get_transactions()['commission'].sum()
        self.summary['Strategy'] = generate_overall_metrics(
            self.results['total_value'].astype(float),
            self._trades,
            self.results['portfolio_value'],
            self.results['daily_returns'],
            capital_base=self._capital_base,
            total_commissions=total_commissions
        )
        # compute benchmark performance metrics if enabled
        if include_benchmarks:
            for benchmark in self._benchmarks:
                self.summary[benchmark] = generate_overall_metrics(
                    self.results[benchmark].astype(float),
                    None,
                    None,
                    self.results['daily_returns']
                )

        # generate trade performance summary
        self.summary_trades = generate_trades_metrics(self._trades, self.results.index)
        # ensure results index is in datetime format
        self.results.index = pd.to_datetime(self.results.index)
        # generate the monthly returns
        self.monthly_returns = generate_monthly_returns(
            self.results['total_value'],
            add_sharpe_ratios=True,
            add_max_drawdowns=True
        )
    # ------------------------------------------ 2 - ORDER-PLACEMENT INTERFACE ------------------------------------------ #
    # order placement methods
    def order(self, asset, amount, limit_price=None, stop_price=None, trade_id=None, target=False):
        """
        places a new order for a specified asset with a given amount.

        parameters:
        - asset: the asset to trade.
        - amount: number of shares to buy/sell.
        - limit_price: price for a Limit Order (default: None).
        - stop_price: price for a Stop Order (default: None).
        - trade_id: optional trade identifier.
        - target: if True, amount represents a target position.

        the order type is determined based on the provided price parameters.
        """
        if limit_price is None and stop_price is None:
            order = MarketOrder(asset, amount, created_at=self.current_bar, unit='shares', trade_id=trade_id, target=target)
        elif limit_price is not None and stop_price is None:
            order = LimitOrder(asset, amount, limit_price, created_at=self.current_bar, unit='shares', trade_id=trade_id, target=target)
        elif limit_price is None and stop_price is not None:
            order = StopOrder(asset, amount, stop_price, created_at=self.current_bar, unit='shares', trade_id=trade_id, target=target)
        else:
            raise NotImplemented  # Stop-Limit orders are not implemented

        self.submit_order(order)  # add the order to the order list

    def order_value(self, asset, value, limit_price=None, stop_price=None, trade_id=None, target=False):
        """
        places an order based on a specified trade value instead of shares.

        parameters:
        - value: the total value ($) to invest in the asset.
        - other parameters are the same as in the `order` method.
        """
        if limit_price is None and stop_price is None:
            order = MarketOrder(asset, value, created_at=self.current_bar, unit='value', trade_id=trade_id, target=target)
        elif limit_price is not None and stop_price is None:
            order = LimitOrder(asset, value, limit_price, created_at=self.current_bar, unit='value', trade_id=trade_id, target=target)
        elif limit_price is None and stop_price is not None:
            order = StopOrder(asset, value, stop_price, created_at=self.current_bar, unit='value', trade_id=trade_id, target=target)
        else:
            raise NotImplemented  # Stop-Limit orders are not implemented

        self.submit_order(order)  # add the order to the order list

    def order_percent(self, asset, percent, limit_price=None, stop_price=None, trade_id=None, target=False):
        """
        places an order based on a percentage (%) of the portfolio value.

        Parameters:
        - percent: the percentage of the portfolio to allocate to the asset.
        - other parameters are the same as in the `order` method.
        """
        if limit_price is None and stop_price is None:
            order = MarketOrder(asset, percent, created_at=self.current_bar, unit='percent', trade_id=trade_id, target=target)
        elif limit_price is not None and stop_price is None:
            order = LimitOrder(asset, percent, limit_price, created_at=self.current_bar, unit='percent', trade_id=trade_id, target=target)
        elif limit_price is None and stop_price is not None:
            order = StopOrder(asset, percent, stop_price, created_at=self.current_bar, unit='percent', trade_id=trade_id, target=target)
        else:
            raise NotImplemented  # Stop-Limit orders are not implemented

        self.submit_order(order)

    def order_target(self, asset, target, limit_price=None, stop_price=None, trade_id=None):
        """
        adjusts an existing position to a target number of shares.

        parameters:
        - target: the desired number of shares to hold.
        """
        self.order(asset, target, limit_price, stop_price, trade_id, target=True)

    def order_target_value(self, asset, target, limit_price=None, stop_price=None, trade_id=None):
        """
        adjusts an existing position to a target value ($).

        parameters:
        - target: the desired value to allocate to the asset.
        """
        if target == 0:
            self.order_target(asset, 0, limit_price, stop_price, trade_id)
            return
        self.order_value(asset, target, limit_price, stop_price, trade_id, target=True)

    def order_target_percent(self, asset, target, limit_price=None, stop_price=None, trade_id=None):
        """
        adjusts an existing position to a target percentage of the portfolio.

        parameters:
        - target: the desired percentage of the portfolio to allocate to the asset.
        """
        if target == 0:
            self.order_target(asset, 0, limit_price, stop_price, trade_id)
            return
        self.order_percent(asset, target, limit_price, stop_price, trade_id, target=True)

    # order mangement methods
    def get_orders(self):
        """ 
        returns the list of all current orders.
        """
        return self._orders

    def clear_orders(self, asset=None):
        """
        removes all orders. if an asset is specified, only orders for that 
        asset are removed.

        parameters:
        - asset: the specific asset whose orders should be cleared 
            (default: None, clears all orders).
        """
        if asset is None:
            self._orders = []  # clear all orders
        else:
            # remove only orders for the given asset
            self._orders = [o for o in self._orders if o.asset != asset]  

    def remove_order(self, order: Order):
        """
        removes a specific order from the order list.

        parameters:
        - order: The order instance to be removed.
        """
        self._orders.remove(order)

    def submit_order(self, order: Order):
        """
        adds a new order to the list of active orders.

        parameters:
        - order: the order instance to be added.
        """
        self._orders.append(order)    
    # ------------------------------------------ 3 - TRANSACTIONS & POSITIONS INTERFACE ------------------------------------------ #
    # transaction management
    def add_transaction(self, trade_id, bar, asset, amount, price, total_value, order_id, commission=0.0):
        """
        records a new transaction in the _transactions DataFrame.

        parameters:
        - trade_id: unique identifier for the trade.
        - bar: timestamp of when the transaction occurred.
        - asset: the asset being bought or sold.
        - amount: the number of shares traded.
        - price: execution price of the transaction.
        - total_value: total value of the trade (amount * price).
        - order_id: ID of the order that triggered this transaction.
        - commission: commission charged for this transaction.
        """
        self._transactions.loc[len(self._transactions)] = [
            trade_id, bar, asset, amount, price, total_value, order_id, commission
        ]

    def get_transactions(self, bar=None):
        """
        retrieves transaction records.

        parameters:
        - bar: if specified, returns transactions that occurred on that specific bar (date).
                if None, returns all transactions.

        returns:
        - a DataFrame containing transaction records.
        """
        if bar is None:
            return self._transactions  # Return all transactions
        else:
            df = self._transactions
            return df[df['bar'] == bar]  # Filter transactions by bar

    def get_latest_transaction(self, asset):
        """
        retrieves the most recent transaction for a given asset.

        parameters:
        - asset: the asset for which to retrieve the latest transaction.

        returns:
        - a single row from the _transactions DataFrame representing the latest trade for the asset.
        """
        return self._transactions[self._transactions['asset'] == asset].iloc[-1]
    
    # position management
    def get_position(self, asset):
        """ 
        returns the current position (total number of shares held) for a specific asset.

        parameters:
        - asset: the asset for which to retrieve the position.

        returns:
        - the net sum of all transactions related to the given asset.
        a positive value indicates a long position, while a negative value indicates a short position.
        """
        return self._transactions[self._transactions['asset'] == asset]['amount'].sum()

    def get_positions(self):
        """
        returns the current positions for all assets.

        returns:
        - a Series where the index is the asset and the value is the net number of shares held.
        """
        return self._transactions[['asset', 'amount']].groupby('asset').sum()['amount']
    # ------------------------------------------ 4 - DATA ACCESS CONTROL ------------------------------------------ #
    # data restriction to prevent look-ahead bias
    def restrict_data(self, full_data: pd.DataFrame) -> tuple:
        """
        restricting the dataset to the CURRENT BAR to prevent look-ahead bias.
        this ensures that decisions are based only on information available at the time.


        parameters:
            - full_data: DataFrame containing the entire dataset with MultiIndex (date, ticker)

        returns:
            - current_data: DataFrame containing data up to the previous bar
            - close: DataFrame containing closing prices from the previous bar
            - open_prices: Series containing open prices for the current bar
        """
        # extract open prices for the current bar. these prices will be used to simulate order execution.
        open_prices = full_data.loc[self.current_bar, (slice(None), 'Open')]  # slice here to get all 'Open' prices from the tickers
        open_prices.index = open_prices.index.get_level_values(0)  # flatten MultiIndex for easier access

        # if there is no previous bar (i.e., this is the first iteration), return early.
        # since we don't have prior close prices yet, return None for current_data and close.
        if self.previous_bar is None:
            return None, None, open_prices

        # extract the closing prices from the previous bar, which are used for signal calculations.
        close = full_data.loc[self.previous_bar]

        # restrict the dataset to only include data up to (and including) the previous bar.
        # this ensures that the strategy cannot access future data when making decisions.
        current_data = full_data.loc[:self.previous_bar]

        return current_data, close, open_prices
    # ------------------------------------------ 5 - PERSISTENCE AND VISUALIZATION ------------------------------------------ #
    def plot(self, benchmark=None, benchmark_label=None, save_to=None):
        """
        plots the strategy's total value and drawdown, comparing it to a benchmark.

        parameters:
        - benchmark: The benchmark to compare against (default: first in self._benchmarks).
        - benchmark_label: Label for the benchmark in the plot (default: benchmark name).
        - save_to: File path to save the plot (default: None, displays plot).
        """
        # use the first benchmark if none is provided
        if benchmark is None:
            benchmark = self._benchmarks[0]
        # use benchmark name as label if no custom label is provided
        if benchmark_label is None:
            benchmark_label = benchmark
        # generate the performance plot
        plot(
            self.results['total_value'],
            self.results['drawdown'],
            self.results[benchmark],
            self.results[f'{benchmark}_drawdown'],
            benchmark_label,
            save_to=save_to
        )

    def to_pickle(self, path):
        """
        saves the strategy object to a file using pickle
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read_pickle(path):
        """
        loads a saved strategy object from a pickle file
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    # ------------------------------------------ 6 - PERFORMANCE METRIC HELPERS ------------------------------------------ #              
    @property
    def num_days(self) -> int:
        """
        returns the number of trading days recorded in the results
        """
        return len(self.results)

    @property
    def previous_total_value(self) -> float:
        """
        returns the total portfolio value from the previous day or the initial capital if none exist
        """
        if self.num_days == 0:
            return self._capital_base
        return self.results.loc[self.results.index[-1], 'total_value']

    @property
    def daily_returns_series(self) -> pd.Series:
        """
        returns the daily returns up to the current trading day
        """
        return self.results.loc[:self.current_bar.date(), 'daily_returns']

    @property
    def portfolio_value_series(self) -> pd.Series:
        """
        returns the portfolio value history up to the current trading day
        """
        return self.results.loc[:self.current_bar.date(), 'portfolio_value']

    @property
    def total_value_series(self) -> pd.Series:
        """
        returns the total portfolio value history up to the current trading day
        """
        return self.results.loc[:self.current_bar.date(), 'total_value']

    def benchmark_value_series(self, benchmark) -> pd.Series:
        """
        returns the benchmark value history up to the current trading day
        """
        return self.results.loc[:self.current_bar.date(), benchmark]

