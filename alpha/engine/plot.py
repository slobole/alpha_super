import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


def plot(
        strategy_total_value: pd.Series,
        strategy_drawdown: pd.Series = None,
        benchmark_total_value: pd.Series = None,
        benchmark_drawdown: pd.Series = None,
        benchmark_label: str = 'Benchmark',
        strategy_label: str = 'Strategy',
        additional_returns: pd.DataFrame = None,
        additional_drawdowns: pd.DataFrame = None,
        save_to: str = None,
        to_web: bool = True,
        dpi: int = 150,
        alpha_additional: float = 0.2,
        colors=None,
        vertical_lines=(),
        ylims=(),
        use_log_scale=True
):
    """
    plots strategy and benchmark performance, drawdowns, and annual returns.

    parameters:
    - strategy_total_value: Series representing the strategy's cumulative returns.
    - strategy_drawdown: Series representing the strategy's drawdowns (optional).
    - benchmark_total_value: Series representing the benchmark's cumulative returns (optional).
    - benchmark_drawdown: Series representing the benchmark's drawdowns (optional).
    - benchmark_label: Label for the benchmark in the plot.
    - strategy_label: Label for the strategy in the plot.
    - additional_returns: DataFrame of additional return series to plot (optional).
    - additional_drawdowns: DataFrame of additional drawdown series to plot (optional).
    - save_to: File path to save the plot (optional).
    - to_web: Whether to optimize the image for web display (default: True).
    - dpi: Resolution of the saved image (default: 150).
    - alpha_additional: Transparency level for additional return lines (default: 0.2).
    - colors: Tuple of two colors for strategy and benchmark plots (optional).
    - vertical_lines: List of timestamps for vertical lines (optional).
    - ylims: Custom y-axis limits for the total return plot (optional).
    - use_log_scale: Whether to use a logarithmic scale for total returns (default: True).
    """

    # Set global plot styles
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5

    # Define colors for strategy and benchmark
    if colors is None:
        color1 = '#6833C9'  # Default strategy color
        color2 = '#FFA870'
        color2 = "#000000"  # Default benchmark color
    else:
        color1, color2 = colors

    # Create the figure layout with three subplots
    fig = plt.figure(figsize=(12, 12))
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, height_ratios=[2, 1],subplot_spec=outer_gs[0], hspace=0.05
    )

    ax1 = fig.add_subplot(inner_gs[0])  # Total return plot
    ax2 = fig.add_subplot(inner_gs[1])  # Drawdown plot
    ax3 = fig.add_subplot(outer_gs[1])  # Annual return bar chart

    # Normalize strategy and benchmark returns
    s_ret = strategy_total_value / strategy_total_value.iloc[0]
    peak_s_ret = s_ret.cummax()
    sy = generate_yearly_returns(strategy_total_value)

    if benchmark_total_value is not None:
        b_ret = benchmark_total_value / benchmark_total_value.iloc[0]
        by = generate_yearly_returns(benchmark_total_value)
        peak_b_ret = b_ret.cummax()

    # --- Total Return Plot (ax1) ---
    if benchmark_total_value is not None:
        ax1.plot(b_ret.index, b_ret, label=benchmark_label, color=color2, zorder=10)
        ax1.fill_between(b_ret.index, peak_b_ret, b_ret, where=(peak_b_ret > b_ret), color='red', alpha=0.5, zorder=6)

    ax1.plot(s_ret.index, s_ret, label=strategy_label, color=color1, zorder=10)
    ax1.fill_between(s_ret.index, peak_s_ret, s_ret, where=(peak_s_ret > s_ret), color='red', alpha=0.5, zorder=6)

    # Plot additional return series if provided
    max_vals = []
    if additional_returns is not None:
        for column in additional_returns.columns:
            line = additional_returns[column]
            ax1.plot(line.index, line / line.iloc[0], alpha=alpha_additional, linewidth=0.5, zorder=5)
            max_vals.append((line / line.iloc[0]).max())

    # Configure log scale if enabled
    if use_log_scale:
        ax1.set_yscale('log')

    ax1.set_xlim(min(s_ret.index), max(s_ret.index))
    ax1.set_xticklabels([])
    ax1.set_ylabel('Total Return (log scale)' if use_log_scale else 'Total Return')
    ax1.legend()

    # Format y-axis labels
    ax1.yaxis.set_major_formatter(FuncFormatter(percentage_major_formatter))
    ax1.yaxis.set_minor_formatter(FuncFormatter(percentage_minor_formatter))

    # Adjust tick marks based on returns
    major_ticks = [0.5] + np.arange(1, 7).tolist()
    minor_ticks = np.arange(1, int(max(s_ret.max(), b_ret.max())) + 1)
    if max_vals:
        minor_ticks = np.arange(1, int(max(max(max_vals), max(s_ret.max(), b_ret.max()))) + 1)

    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='#e6e6e6')
    ax1.grid(which='minor', linestyle='-', linewidth='0.5', color='#e6e6e6')

    if len(ylims) > 0:
        ax1.set_ylim(ylims)

    # --- Drawdown Plot (ax2) ---
    if strategy_drawdown is None:
        cumulative_max = strategy_total_value.cummax()
        strategy_drawdown = (strategy_total_value - cumulative_max) / cumulative_max
    if benchmark_drawdown is None and benchmark_total_value is not None:
        cumulative_max = benchmark_total_value.cummax()
        benchmark_drawdown = (benchmark_total_value - cumulative_max) / cumulative_max

    if benchmark_drawdown is not None:
        ax2.fill_between(benchmark_drawdown.index, 0, benchmark_drawdown, color=color2, alpha=0.1, zorder=1)
        ax2.plot(benchmark_drawdown.index, benchmark_drawdown, color=color2, zorder=10, linewidth=0.5)

    if strategy_drawdown is not None:
        ax2.fill_between(strategy_drawdown.index, 0, strategy_drawdown, color=color1, alpha=0.1, zorder=2)
        ax2.plot(strategy_drawdown.index, strategy_drawdown, color=color1, zorder=10, linewidth=0.5)

    if additional_drawdowns is not None:
        for column in additional_drawdowns.columns:
            line = additional_drawdowns[column]
            ax2.plot(line.index, line, alpha=0.2, linewidth=0.5, zorder=5)

    ax2.set_xlim(min(strategy_total_value.index), max(strategy_total_value.index))
    ax2.set_ylabel('Drawdown')
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='#e6e6e6')
    ax2.yaxis.set_major_formatter(FuncFormatter(percentage_major_formatter_2))

    # --- Annual Return Bar Chart (ax3) ---
    bar_width = 0.35
    index = np.arange(len(sy))

    bar1 = ax3.bar(index, sy, bar_width, label=strategy_label, color=color1, zorder=10)
    if benchmark_total_value is not None:
        bar2 = ax3.bar(index + bar_width, by, bar_width, label=benchmark_label, color=color2, zorder=10)

    ax3.set_ylabel('Annual Return (%)')
    ax3.set_xticks(index + bar_width / 2)
    ax3.set_xticklabels(sy.index.year, rotation=90)
    ax3.set_xlim(-bar_width, len(sy) - bar_width)
    ax3.yaxis.set_major_formatter(FuncFormatter(percentage_major_formatter_2))
    ax3.grid(which='major', linestyle='-', linewidth='0.5', color='#e6e6e6', axis='y', zorder=0)
    

    # --- Set the style ---
    positions = np.arange(len(sy))
    grid_positions = np.append(positions, positions[-1] + 1)
    for pos in grid_positions:
        ax3.axvline(x=pos - bar_width, color='#e6e6e6', linestyle='-', linewidth=0.5, zorder=0)

    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#E6E6E6')
            spine.set_linewidth(0.5)

        ax.tick_params(axis='x', colors='#666666')
        ax.tick_params(axis='y', colors='#666666')
        ax.tick_params(axis='y', which='both', color='#e6e6e6', labelsize=6)
        ax.tick_params(axis='x', which='both', color='#e6e6e6')
        if ax == ax3:
            ax.tick_params(axis='x', which='both', color='#e6e6e6', labelsize=8)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('left')

        # Plot the vertical lines
        for line in vertical_lines:
            ax.axvline(line, color='red', linestyle='--', linewidth=1)

    # Add the labels
    s_tot_ret = strategy_total_value.iloc[-1] / strategy_total_value.iloc[0] - 1
    strategy_line_label = f"+{s_tot_ret:,.0%}"
    add_label(ax1, ax1.lines[1], strategy_line_label, color1, 'white', zorder=20)

    if benchmark_total_value is not None:
        b_tot_ret = benchmark_total_value.iloc[-1] / benchmark_total_value.iloc[0] - 1
        benchmark_line_label = f"+{b_tot_ret:,.0%}"
        add_label(ax1, ax1.lines[0], benchmark_line_label, color2, 'white')

    # --- Save or Show the Plot ---
    if save_to:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')

    plt.show()


def generate_yearly_returns(series):
    """Computes annual returns based on the last value of each year.

    Parameters:
    - series: Series representing total portfolio value over time.

    Returns:
    - A Series with annual percentage returns.
    """
    sy = series.resample('YE').last()  # Get the last value of each year

    # Ensure the first year includes the initial value
    if sy.index[0] != series.index[0]:
        date0 = pd.Timestamp(year=sy.index[0].year - 1, month=sy.index[0].month, day=sy.index[0].day)
        sy.loc[date0] = series.iloc[0]
        sy = sy.sort_index()

    return sy.pct_change(fill_method=None).dropna()  # Compute percentage change and drop NaN values


def add_label(ax, line, label, bg_color, font_color='black', zorder=5):
    """Adds a label with an arrow at the end of a line plot.

    Parameters:
    - ax: The axis on which to annotate.
    - line: The line to be labeled.
    - label: The text to display.
    - bg_color: Background color of the label.
    - font_color: Color of the text (default: black).
    - zorder: Layer order for overlapping elements (default: 5).
    """
    x_data, y_data = line.get_data()  # Extract x and y coordinates of the line

    ax.annotate(
        label,
        xy=(x_data[-1], y_data[-1]),  # Position label at the last point of the line
        xytext=(3, 0),  # Offset label slightly to the right
        textcoords='offset points',
        ha='left',
        va='center',
        fontsize=8,
        color=font_color,
        bbox=dict(boxstyle="round,pad=0.4", edgecolor='none', facecolor=bg_color),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=bg_color, lw=2),
        zorder=zorder
    )


def percentage_major_formatter(x, pos):
    """Formats major y-axis ticks for total return plots as percentages.
    """
    return f'{(x - 1) * 100:.0f}%'


def percentage_major_formatter_2(x, pos):
    """Formats major y-axis ticks for drawdown and annual return plots
    as percentages.
    """
    return f'{x * 100:.0f}%'


def percentage_minor_formatter(x, pos):
    """Formats minor y-axis ticks. Returns an empty string to keep
    the minor ticks clean.
    """
    return ''
