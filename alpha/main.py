import sys

def main():
    import norgatedata

    priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN 
    padding_setting = norgatedata.PaddingType.NONE   
    symbol = 'AAPL'
    start_date = '1990-01-01'
    timeseriesformat = 'numpy-recarray'

    # Provides data on GOOG from 1990 until today in 
    # a NumPy recarray format, with explicitly set stock price 
    # adjustment and padding settings
    pricedata_recarray = norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting = priceadjust,
        padding_setting = padding_setting,
        start_date = start_date,
        timeseriesformat=timeseriesformat,
    )
    return pricedata_recarray


if __name__ == "__main__":
    df = main()
