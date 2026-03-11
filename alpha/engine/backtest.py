# alpha/engine/backtest.py

"""
backtest.py
------------
this module implements the run_daily() method, responsible for the main loop of the simulation. 
it initializes the strategy, and then iterates throughout the calendar, calling the strategy methods that process 
orders and update results. Finally, at the end, the method summarizes the overall results.

------------
The run_daily() method:
The run_daily method is responsible for executing a systematic trading strategy on a daily basis using historical 
price data. It takes a Strategy object, a pricing_data DataFrame containing market data, and an optional calendar 
that defines the trading days to iterate through. If no calendar is provided, the function defaults to using the 
index of pricing_data. The function follows a structured loop where, for each trading day, it restricts data to 
prevent look-ahead bias, calculates trading signals, processes orders, and updates performance metrics. 
Additionally, it includes a progress tracker that displays cumulative and annualized returns, helping monitor 
the strategy's performance in real time.

At the start of the simulation, the function precomputes trading signals for the entire dataset using strategy.
compute_signals(), ensuring efficient execution during iteration. The first iteration requires special handling 
because there is no prior bar to reference, so it initializes performance tracking before proceeding. 
For each subsequent day in the calendar, the function follows a well-defined sequence: it restricts data to 
the current date, executes the strategy's logic via iterate(), processes any resulting orders, and updates 
key performance metrics. 

Finally, once all days have been processed, the function finalizes the simulation and 
generates a summary report. This method is critical for backtesting strategies in a controlled, systematic manner,
ensuring that trading decisions are made under realistic market conditions.

*** IMPORTANT NOTE ***:
-------------
In our framework, it is essential to note that the iterate and process_orders methods are executed at the 
market open. This is why we call the iterate method while providing open prices, ensuring that trading decisions 
are based on the prices available at the start of the session. Meanwhile, the update_metrics method captures and 
records key performance indicators at the end of the trading day, reflecting the day's final market conditions.

-------------
Pricing Data: pd_DataFrame table:

Index: The index must consist of dates, representing the timestamps for the simulation.
Columns: The columns must be organized into a multi-level structure, where:
The first level represents the symbols (i.e., asset tickers used in the simulation).
The second level contains the associated price data and features (e.g., open, high, low, close, volume, turnover, etc.).
At a minimum, each symbol must include OHLC (Open, High, Low, Close) prices to ensure the strategy has essential market data.
"""

from tqdm import tqdm
import pandas as pd
from alpha.engine.strategy import Strategy


def run_daily(strategy: Strategy, pricing_data: pd.DataFrame, calendar: pd.DatetimeIndex = None, 
              show_progress: bool = True):
    if calendar is None:
        calendar = pricing_data.index # use pricing data index if no custom calendar is provided

    # 1. compute trading signals for the entire dataset at the start of the simulation.
    # in backtests, signals are typically precomputed to avoid recalculating them on every iteration.
    print('precomputing signals...')
    full_data = strategy.compute_signals(pricing_data)
    if strategy.enable_signal_audit:
        strategy.audit_signals(pricing_data, full_data)

    # set the initial 'previous_bar' if a prior data point exists, ensuring continuity in calculations.
    # this ensures that the first iteration has a valid previous bar for reference.
    if pricing_data.index.get_loc(calendar[0]) != 0:
        strategy.previous_bar = pricing_data.index[pricing_data.index.get_loc(calendar[0]) - 1]

    # initialize progress tracking
    print('starting backtest...')
    pbar = tqdm(calendar, desc="backtest") if show_progress else calendar

    # 2. iterate through the calendar, processing each bar in the pricing data.
    for i, bar in enumerate(pbar):
        # the first bar iteration requires special handling, as there's no previous bar yet.
        if strategy.previous_bar is None:
            strategy.current_bar = bar  # set current bar
            strategy.update_metrics(pricing_data, calendar[0])  # initialize performance metrics
            strategy.previous_bar = bar  # store the current bar as previous for the next iteration
            continue  # skip further processing in this iteration + no trade should be placed !!

        # update the current bar in the simulation
        strategy.current_bar = bar

        # 2.1. restrict available data to the current bar to prevent look-ahead bias
        # close -> close price of yesterday ! _open_ -> open price of today ! to prevent look-ahead bias !!
        current_data, close, open_ = strategy.restrict_data(full_data)

        # 2.2. compute the logic of trading decisions and place orders based on restricted data (up to previous bar)
        strategy.iterate(current_data, close, open_)

        # 2.3. process any orders that were placed in the iteration
        strategy.process_orders(pricing_data)

        # 2.4. update performance metrics for this iteration
        strategy.update_metrics(pricing_data, calendar[0])

        # 2.5. store the current bar as the previous bar for the next iteration
        strategy.previous_bar = bar

        # 2.6. update progress bar with performance statistics
        total_return = strategy.total_value / strategy._capital_base - 1
        num_days = strategy.num_days
        annualized_return = (1 + total_return) ** (252 / (num_days + 1)) - 1  # CAGR
        if show_progress:
            pbar.set_postfix(
                bar=bar.strftime('%Y-%m-%d'),
                total_return=f'{total_return:.1%}',
                annualized_return=f'{annualized_return:.1%}'
            )

    # 3.0 finalize the strategy run and generate performance summaries
    strategy.finalize(full_data)
    strategy.summarize()
