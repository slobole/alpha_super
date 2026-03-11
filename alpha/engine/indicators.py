import pandas as pd

def qp_indicator(close, window=3, lookback_years=5):
    try:
        rets = close.pct_change(periods=window, fill_method=None)
    except ValueError as e:
        return pd.Series(index=close.index)

    def rolling_rank(x):
        return (x.rank(pct=True).iloc[-1])

    rolling_rank = rets.rolling(window=lookback_years * 252).apply(rolling_rank, raw=False)
    rolling_down = rets.rolling(window=lookback_years * 252).apply(lambda x: (x <= 0).sum())
    prob_down = rolling_down / (lookback_years * 252)
    prob_up = 1 - prob_down

    idx_down = rets[rets <= 0].index
    idx_up = rets[rets > 0].index

    qp = pd.Series(index=close.index)
    qp.loc[idx_down] = rolling_rank / prob_down
    qp.loc[idx_up] = (1 - rolling_rank) / prob_up
    qp *= 100
    return qp


def dv2_indicator(close, high, low, length=126):
    """
    the Varadi Oscillator (DVO) is a leading indicator first proposed by David Varadi and originally aimed to reduce the 
    influence of the trend component in oscillators. 
    the DVO can be described as a rolling percent rank of detrended prices over a particular lookback period.
    here we are calculating the 2 period DVO.
    """
    hl2 = (high + low) / 2
    dv1 = (close / hl2) - 1
    dv = dv1.rolling(window=2).mean()
    return dv.rolling(window=length).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]) * 100

