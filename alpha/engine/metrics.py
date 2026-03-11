# alpha/engine/metrics.py

"""
metrics.py
-----------
provides utility functions for analyzing strategy performance after a simulation run.

includes standard risk-adjusted return metrics and statistical summaries of trades
and drawdowns. used internally by the strategy class during summarization.

overview:
* generate_trades - identifies and summarizes individual trades from transaction data.
* generate_drawdowns - calculates drawdowns based on portfolio value fluctuations.
* generate_overall_metrics - computes key performance metrics such as returns, volatility, and the Sharpe ratio.
* generate_trades_metrics - aggregates trade statistics, including win/loss ratio and average trade return.
"""
import numpy as np
import pandas as pd

def sharpe_ratio(daily_returns, portfolio_value=None, days_in_year=252):
    """
    computes the annualized Sharpe ratio, which measures risk-adjusted 
    returns.

    formula (risk free is assumed to be 0):
    annual_sr = (mean(daily_returns) / std(daily_returns)) * sqrt(days_in_year)

    parameters:
    - daily_returns: Series of daily returns.
    - portfolio_value: Series of portfolio values (optional, used to exclude 
        zero-value periods).
    - days_in_year: Number of trading days in a year (default: 252).

    returns:
    - the Sharpe ratio, which represents the strategy's return per unit of risk.

    what it means ? for example: for a Sharpe ratio of 1.5, you earned 1.5 units of return for every unit of risk taken.
    """

    if portfolio_value is None:
        rets = daily_returns  # use all daily returns if portfolio value is not provided
    else:
        # exclude periods where portfolio value or returns are zero to avoid skewed results ( excluding dead periods )
        pv = portfolio_value != 0
        re = daily_returns != 0
        rets = daily_returns[(re[re == True] | pv[pv == True]).index]

    # calculate the mean and standard deviation of returns
    mu, std = rets.mean(), rets.std()
    # Sharpe
    if std == 0:
        return np.nan
    return (mu / std) * np.sqrt(days_in_year)

def sortino_ratio():
    raise NotImplemented

def generate_trades(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    groups transactions into trades based on trade_id and calculates 
    key trade statistics.

    parameters:
    - transactions: DataFrame containing all executed transactions.

    returns:
    - a DataFrame summarizing closed trades, including start and end times, 
    assets involved, initial capital, profit, duration, and return.

    example:
    transactions:
        | trade_id | bar        | asset | amount | price | total_value | order_id |
        | --------- | ---------- | ----- | ------ | ----- | ------------ | --------- |
        | 1         | 2024-01-02 | AAPL  | 10     | 100   | 1000         | 101       |
        | 1         | 2024-01-10 | AAPL  | -10    | 110   | -1100        | 102       |
    results:
        | trade_id | start      | end        | assets | capital | profit | duration | return |
        | --------- | ---------- | ---------- | ------ | ------- | ------ | -------- | ------ |
        | 1         | 2024-01-02 | 2024-01-10 | AAPL   | 1000    | 100    | 8 days   | 10%    |
    """
    def concat_str(series):
        """
        converts a series of asset names into a comma-separated string 
        of unique values.
        this ensures that multi-asset trades are represented correctly.
        """
        v = list(set(series.values.tolist()))  # Extract unique asset names
        return ', '.join(v)  # Convert to comma-separated string

    # define aggregation functions for trade grouping
    f = {
        'bar': ['min', 'max'],  # get the start and end bars (timestamps)
        'asset': concat_str,  # combine asset names into a single string
        'total_value': ['max', 'sum'],  # capture max capital and total profit
        'amount': 'sum',  # sum the trade amount to determine if trade is closed
        'commission': 'sum'  # total commission for the trade
    }
    # group transactions by trade_id and apply aggregation
    trades = transactions[['amount', 'trade_id', 'bar', 'asset', 'total_value', 'commission']]
    trades = trades.groupby('trade_id').agg(f)
    # rename columns for clarity
    trades.columns = ['start', 'end', 'assets', 'capital', 'profit', 'amount', 'commission']
    # convert profit values to negative (since sell transactions have negative total_value)
    trades['profit'] = -trades['profit'] - trades['commission']
    # filter only closed trades for final output
    trades = trades[trades['amount'] == 0]
    trades = trades[['start', 'end', 'assets', 'capital', 'profit', 'commission']]
    # calculate trade duration (time between opening and closing)
    trades['duration'] = trades['end'] - trades['start']
    # compute trade return as profit divided by initial capital
    trades['return'] = trades['profit'] / trades['capital']
    return trades

def generate_drawdowns(drawdown_series: pd.Series) -> pd.DataFrame:
    """
    identifies and groups drawdown periods from a time series of 
    daily drawdowns.

    parameters:
    - drawdown_series: A Series containing daily drawdown values.

    returns:
    - a DataFrame with drawdown periods, including start and end dates,
      duration, and maximum drawdown value.
    """
    def split_nonzero_subseries(series):
        """
        splits the drawdown series into continuous non-zero drawdown periods.
        this method scans the series and groups consecutive non-zero values,
        effectively identifying individual drawdown periods.

        returns:
        - a list of Series, each representing a separate drawdown period.

        example:
        drawdown_series:
        2024-01-01    0      ← Not underwater
        2024-01-02   -1%
        2024-01-03   -3%
        2024-01-04   -2%     ← Drawdown group #1
        2024-01-05    0      ← Ends first drawdown
        2024-01-06    0
        2024-01-07   -4%
        2024-01-08   -5%
        2024-01-09   -1%     ← Drawdown group #2
        2024-01-10    0

        output:
        | start      | end        | duration | value |
        | ---------- | ---------- | -------- | ----- |
        | 2024-01-02 | 2024-01-04 | 3        | -0.03 |
        | 2024-01-07 | 2024-01-09 | 3        | -0.05 |
        """
        non_zero_groups = []
        current_group = []

        for index, value in series.items():
            if value != 0:
                current_group.append(index)  # add index to current drawdown group
            else:
                if current_group:
                    non_zero_groups.append(series[current_group])  # save completed drawdown period
                    current_group = []

        # append the last drawdown period if one exists
        if current_group:
            non_zero_groups.append(series[current_group])

        return non_zero_groups

    # identify and split continuous drawdown periods
    dds = split_nonzero_subseries(drawdown_series)
    # create an empty DataFrame to store drawdown periods
    drawdowns = pd.DataFrame(columns=['start', 'end', 'duration', 'value'])
    # populate the DataFrame with drawdown start, end, duration, and minimum drawdown value
    for dd in dds:
        drawdowns.loc[len(drawdowns)] = [dd.index[0], dd.index[-1], len(dd), dd.min()]

    # convert 'start' and 'end' columns to datetime format
    for column in ['start', 'end']:
        drawdowns[column] = pd.to_datetime(drawdowns[column])

    return drawdowns

def generate_overall_metrics(total_value: pd.Series, trades: pd.DataFrame = None, portfolio_value: pd.Series = None,
                             series_to_correlate: pd.Series = None, capital_base: float = None, days_in_year: int = 252,
                             total_commissions: float = None) -> pd.Series:
    """
    computes overall performance metrics for a trading strategy.

    parameters:
    - total_value: Series representing the portfolio's total value over time.
    - trades: DataFrame of closed trades (optional, used for exposure time calculation).
    - portfolio_value: Series representing the portfolio's value excluding cash 
        (optional, used for Sharpe ratio).
    - series_to_correlate: benchmark series for correlation analysis (optional).
    - capital_base: initial portfolio value (optional, defaults to first value 
        of total_value).
    - days_in_year: number of trading days in a year (default: 252).

    returns:
    - a Series containing key performance metrics.
    """
    s = pd.Series()  # initialize the results series
    # store simulation start and end dates
    s.loc['Start'] = total_value.index[0]
    s.loc['End'] = total_value.index[-1]
    s.loc['Duration [days]'] = len(total_value)
    # compute exposure time (percentage of time the portfolio was active)
    if trades is not None:
        num_days = 0
        calendar = pd.to_datetime(total_value.index)
        for date in calendar:
            if ((date >= trades['start']) & (date <= trades['end'])).any():
                num_days += 1
        exposure_time = num_days / s.loc['Duration [days]']
    else:
        exposure_time = 1  # assume 100% exposure if no trade data is available

    s.loc['Exposure Time [%]'] = exposure_time * 100  # convert to percentage
    # determine initial capital base if not provided
    if capital_base is None:
        capital_base = total_value.iloc[0]

    # store key portfolio values
    s.loc['Start [$]'] = capital_base
    s.loc['Final [$]'] = total_value.iloc[-1]
    if total_commissions is not None:
        s.loc['Total Commissions [$]'] = total_commissions
    s.loc['Peak [$]'] = total_value.max()
    # compute total and annualized returns
    s.loc['Return [%]'] = (total_value.iloc[-1] / capital_base - 1) * 100
    num_days = len(total_value)
    s.loc['Return (Ann.) [%]'] = ((total_value.iloc[-1] / capital_base) ** (days_in_year / num_days) - 1) * 100
    # compute annualized volatility
    daily_rets = total_value.pct_change(fill_method=None)
    s.loc['Volatility (Ann.) [%]'] = daily_rets.std() * np.sqrt(days_in_year) * 100
    # compute Sharpe ratio
    s.loc['Sharpe Ratio'] = sharpe_ratio(daily_rets, portfolio_value, days_in_year=days_in_year)
    # compute exposure-adjusted return (only if exposure time is nonzero)
    if exposure_time == 0:
        s.loc['Exposure-Adjusted Return (Ann.) [%]'] = 0
    else:
        s.loc['Exposure-Adjusted Return (Ann.) [%]'] = s.loc['Return (Ann.) [%]'] / exposure_time

    # compute correlation with a benchmark series (if provided)
    if series_to_correlate is not None:
        pct = daily_rets.dropna()
        idx = pct.index.intersection(series_to_correlate.index)  # align indices
        s.loc['Correlation'] = pct[idx].corr(series_to_correlate[idx])
    else:
        s.loc['Correlation'] = 1  # Default correlation value if no benchmark is given

    # compute drawdown metrics
    running_max = total_value.cummax()
    drawdown = total_value / running_max - 1
    s.loc['Max. Drawdown [%]'] = drawdown.min() * 100
    # generate drawdown statistics
    drawdowns = generate_drawdowns(drawdown)
    s.loc['Avg. Drawdown [%]'] = drawdowns['value'].mean() * 100
    # compute drawdown durations
    u = pd.Timedelta(total_value.index[-1] - total_value.index[0]).resolution_string
    s.loc['Max. Drawdown Duration [days]'] = pd.Timedelta(drawdowns['duration'].max(), unit=u).days
    s.loc['Avg. Drawdown Duration [days]'] = pd.Timedelta(np.ceil(drawdowns['duration'].mean()), unit=u).days
    # count drawdown occurrences
    s.loc['# Drawdowns'] = len(drawdowns)
    s.loc['# Drawdowns / year'] = int(len(drawdowns) / (s.loc['Duration [days]'] / 365))
    # return the computed metrics
    return s

def cross_correlation_matrix(daily_rets: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation matrix of strategy daily returns."""
    return daily_rets.corr()


def diversification_ratio(daily_rets: pd.DataFrame, weights: list[float]) -> float:
    """Compute the diversification ratio of a weighted portfolio.

    Ratio > 1.0 means diversification benefit exists.
    Ratio = 1.0 means perfect correlation (no benefit).
    """
    w = np.array(weights)
    sigmas = daily_rets.std().values
    weighted_sum_vol = (w * sigmas).sum()
    cov = daily_rets.cov().values
    port_vol = np.sqrt(w @ cov @ w)
    return weighted_sum_vol / port_vol if port_vol > 0 else np.nan


def generate_trades_metrics(all_trades: pd.DataFrame, calendar: pd.DatetimeIndex, unit='days') -> pd.DataFrame:
    """
    computes key trade statistics, categorizing them into all trades, 
    winning trades, and losing trades.

    parameters:
    - all_trades: DataFrame containing all closed trades.
    - calendar: DatetimeIndex representing the full simulation period.
    - unit: time unit for trade duration metrics (default: 'days').

    returns:
    - a DataFrame summarizing trade statistics
    """
    df = pd.DataFrame()  # initialize an empty DataFrame to store trade metrics
    
    # define categories for trade analysis: all trades, winning trades, and losing trades
    for trades, title in [
            (all_trades, 'All Trades'),
            (all_trades[all_trades['return'] > 0], 'Winning Trades'),
            (all_trades[all_trades['return'] <= 0], 'Losing Trades')
    ]:
        # skip categories with no trades
        if len(trades) == 0:
            continue

        # basic trade statistics
        df.loc[title, '# Trades'] = len(trades)
        duration_days = (calendar[-1] - calendar[0]).days  # total simulation duration in days
        df.loc[title, '# Trades / year'] = int(len(trades) / (duration_days / 365.25))
        df.loc[title, '# Trades / week'] = round(len(trades) / (duration_days / 7), 1)
        df.loc[title, 'Avg. return / trade [%]'] = trades['return'].mean() * 100  # average trade return
        # record the best or worst trade based on the trade category
        if title == 'All Trades':
            df.loc[title, 'Best / worst trade [%]'] = ''
        elif title == 'Winning Trades':
            df.loc[title, 'Best / worst trade [%]'] = trades['return'].max() * 100
        else:
            df.loc[title, 'Best / worst trade [%]'] = trades['return'].min() * 100

        # trade duration statistics
        df.loc[title, f'Max. Trade Duration [{unit}]'] = trades['duration'].max()
        df.loc[title, f'Avg. Trade Duration [{unit}]'] = trades['duration'].mean()
        # additional metrics for all trades
        if title == 'All Trades':
            win_rate = len(trades[trades['return'] > 0]) / len(trades)  # percentage of winning trades
            gross_profit = trades[trades['return'] > 0]['profit'].sum()  # total profit from winning trades
            gross_loss = trades[trades['return'] < 0]['profit'].sum()  # total loss from losing trades
            avg_win_value = trades[trades['return'] > 0]['profit'].mean()  # average profit per winning trade
            avg_loss_value = trades[trades['return'] < 0]['profit'].mean()  # average loss per losing trade
            avg_win_return = trades[trades['return'] > 0]['return'].mean()  # average return per winning trade
            avg_loss_return = trades[trades['return'] < 0]['return'].mean()  # average return per losing trade
            df.loc[title, 'Win Rate [%]'] = win_rate * 100
            df.loc[title, 'Profit Factor'] = abs(gross_profit / gross_loss)  # profit-to-loss ratio
            # compute win/loss ratio (avoid division by zero)
            if len(trades[trades['return'] < 0]) == 0:
                df.loc[title, 'Win/Loss Ratio'] = np.nan
            else:
                df.loc[title, 'Win/Loss Ratio'] = len(trades[trades['return'] > 0]) / len(trades[trades['return'] < 0])

            df.loc[title, 'Payoff Ratio'] = abs(avg_win_return / avg_loss_return)  # ratio of average win to average loss
            df.loc[title, 'CPC Index'] = df.loc[title, 'Profit Factor'] * win_rate * df.loc[title, 'Win/Loss Ratio']
            df.loc[title, 'Expectancy [$]'] = avg_win_value * win_rate - abs(avg_loss_value) * (1 - win_rate)
            if 'commission' in trades.columns:
                df.loc[title, 'Avg. Commission / trade [$]'] = trades['commission'].mean()

    # convert DataFrame columns to object type to handle mixed data
    df = df.T.astype('O')
    # fill empty values with an empty string for cleaner display
    df = df.fillna('')
    # ensure integer columns are properly formatted
    for row in ['# Trades', '# Trades / year']:
        if row in df.index:
            df.loc[row] = df.loc[row].astype('int')

    # convert trade duration values to integer days if the unit is 'days'
    if unit == 'days':
        df.loc['Avg. Trade Duration [days]'] = df.loc['Avg. Trade Duration [days]'].apply(lambda x: x.days)
        df.loc['Max. Trade Duration [days]'] = df.loc['Max. Trade Duration [days]'].apply(lambda x: x.days)

    return df

def generate_monthly_returns(series: pd.Series, add_max_drawdowns=False, add_calmar_ratios=False, 
                             add_sharpe_ratios=False, benchmark=None) -> pd.DataFrame:
    """
    computes monthly and annual returns, with optional risk metrics.

    parameters:
    - series: Series representing the portfolio's total value over time.
    - add_max_drawdowns: if True, includes the maximum drawdown for each year.
    - add_calmar_ratios: if True, includes the Calmar ratio (Annual Return / Max Drawdown).
    - add_sharpe_ratios: if True, includes the Sharpe ratio for each year.
    - benchmark: optional benchmark series to compare annual returns.

    returns:
    - a DataFrame with years as rows and months as columns, displaying 
        monthly and annual returns.
    """

    def get_annual_max_drawdowns(total_value):
        """
        calculates the maximum drawdown for each year.
        """
        def calculate_max_drawdown(group):
            cum_max = group.cummax()  # Track the highest portfolio value
            drawdowns = (group - cum_max) / cum_max  # Compute drawdown
            return drawdowns.min()  # Return the worst drawdown

        # apply drawdown calculation by year
        if isinstance(total_value.index, pd.DatetimeIndex):
            return total_value.groupby(total_value.index.year).apply(calculate_max_drawdown)
        else:
            return total_value.groupby(total_value.index).apply(calculate_max_drawdown)

    total_value = series  # store the original total value series
    daily_rets = total_value.pct_change(fill_method=None).fillna(0)  # compute daily returns
    # store the initial portfolio value for reference
    initial_val = series.iloc[0]
    # convert index to DateTime format and resample data to get the last value of each month
    series.index = pd.DatetimeIndex(series.index)
    series = series.resample('ME').last()
    # calculate monthly returns (percentage change from the previous month)
    monthly_returns = series.pct_change(fill_method=None)
    monthly_returns.iloc[0] = series.iloc[0] / initial_val - 1  # set first month’s return correctly
    # create a DataFrame with monthly returns, adding year and month columns
    monthly_returns_df = monthly_returns.to_frame(name='monthly_return')
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    # pivot the DataFrame to organize data with years as rows and months as columns
    pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='monthly_return')
    # calculate annual returns and add them as a separate column
    annual_returns = series.resample('YE').last().pct_change(fill_method=None).dropna()
    annual_returns.index = annual_returns.index.year
    pivot_table['Annual Return'] = annual_returns
    # reorder the columns so that months appear in order, followed by the annual return
    pivot_table = pivot_table.reindex(columns=list(range(1, 13)) + ['Annual Return'])
    pivot_table = pivot_table.sort_index(ascending=False)  # Sort by year in descending order
    # compute the first year's return manually (since pct_change() drops the first value)
    first = series.resample('YE').last().iloc[0] / initial_val - 1
    pivot_table.iloc[-1, -1] = first
    # add benchmark annual returns if provided
    if benchmark is not None:
        pivot_table['Benchmark'] = generate_monthly_returns(benchmark)['Annual Return']

    # add maximum drawdowns per year if requested
    if add_max_drawdowns:
        pivot_table['Max Drawdown'] = get_annual_max_drawdowns(total_value)

    # compute Calmar ratio (Annual Return / Max Drawdown) if requested
    if add_calmar_ratios:
        pivot_table['Calmar Ratio'] = pivot_table['Annual Return'] / pivot_table['Max Drawdown'].abs()

    # compute Sharpe ratio per year if requested
    if add_sharpe_ratios:
        for year in pivot_table.index:
            pivot_table.loc[year, 'Sharpe Ratio'] = sharpe_ratio(daily_rets.loc[daily_rets.index.year == year])

    return pivot_table