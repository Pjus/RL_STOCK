import pandas as pd
import numpy as np
import pandas_datareader as pdr

ticker = 'AAPL'
start_date = '2000-01-01'

data = pdr.get_data_yahoo(ticker, start_date)
data['Date'] = data.index
data.index = range(len(data))

COLUMNS_CHART_DATA = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

COLUMNS_TRAINING_DATA = [
    'open_lastclose_ratio', 'high_close_ratio', 'stock_code',
]
