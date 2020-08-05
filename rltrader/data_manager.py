import pandas as pd
import numpy as np
import pandas_datareader as pdr
import indicator
import pandas_datareader as pdr

COLUMNS_CHART_DATA = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

def load_data(ticker, start_date, end_date):
    data = pdr.get_data_yahoo(ticker, start_date, end_date)
    data['Date'] = data.index
    data.index = range(len(data))
    data = data.sort_values('Date')

    return data

def preprocess(data, ver='v1'):
    windows = [5, 10, 20, 60, 120]
    data = indicator.faster_OBV(data)
    for window in windows:
        # MACD
        data['close_ma{}'.format(window)] = data['Close'].rolling(window).mean()
        data['close_ma{}'.format(window)] = data['Close'].rolling(window).mean()
        data['close_ma{}'.format(window)] = data['Close'].rolling(window).mean()

        # DMI PMI ADX
        data = indicator.DMI(data, n=window, n_ADX=window)
        
        # Bollinger Band
        data = indicator.fnBolingerBand(data, window)

        # RSI
        data = indicator.fnRSI(data, window)

        # CCI
        data = indicator.CCI(data, window)

        # EVM
        data = indicator.EVM(data, window)

        # SMA
        data = indicator.SMA(data, window)

        # EWMA
        data = indicator.EWMA(data, window)

        # ROC
        data = indicator.ROC(data, window)

        # forceindex
        data = indicator.ForceIndex(data, window)

    data.dropna(inplace=True)
    return data

def get_train_data(chart_data):
    TRAIN_DATA = data.drop(['Date'], axis=1)
    return chart_data, TRAIN_DATA


if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2001-01-01'
    end_date = '2020-08-01'

    chart_data = load_data(ticker, start_date, end_date)
    preprocessed = preprocess(chart_data)
    train_data = get_train_data(preprocessed)

    









