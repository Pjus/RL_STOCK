import pandas as pd
import numpy as np
import pandas_datareader as pdr
import indicator
import datetime

now = datetime.datetime.now()
nowDate = now.strftime('%Y-%m-%d')

COLUMNS_CHART_DATA = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def load_data(ticker, start_date, end_date=nowDate): 
    data = pdr.get_data_yahoo(ticker, start_date, end_date)
    data['Date'] = data.index
    data.index = range(len(data))
    data = data.sort_values('Date')

    return data

def preprocess(data, ver='v1'):
    windows = [5, 10, 20, 60, 120]
    data = indicator.faster_OBV(data)
    data['log_OBV'] = np.log(data['OBV'])

    for window in windows:
        data['close_ma{}'.format(window)] = data['Close'].rolling(window).mean()
        data['volume_ma{}'.format(window)] = data['Volume'].rolling(window).mean()
        data['close_ma%d_ratio' % window] = (data['Close'] - data['close_ma%d' % window]) / data['close_ma%d' % window]
        data['volume_ma%d_ratio' % window] = (data['Volume'] - data['volume_ma%d' % window]) / data['volume_ma%d' % window]

        # DMI PMI ADX
        data = indicator.DMI(data, n=window, n_ADX=window)
        data['MDI_ADX_ratio{}'.format(window)] = data['MDI_{}'.format(window)] / data['ADX_{}'.format(window)]
        data['PDI_ADX_ratio{}'.format(window)] = data['PDI_{}'.format(window)] / data['ADX_{}'.format(window)]
        
        # Bollinger Band
        data = indicator.fnBolingerBand(data, window)
        data['Bol_upper_close_ratio{}'.format(window)] = data['Bol_upper_{}'.format(window)] / data['close_ma{}'.format(window)]
        data['Bol_lower_close_ratio{}'.format(window)] = data['Bol_lower_{}'.format(window)] / data['close_ma{}'.format(window)]
        
        # RSI
        data = indicator.fnRSI(data, window)
        # CCI
        data = indicator.CCI(data, window)
        # EVM
        data = indicator.EVM(data, window)
        # EWMA
        data = indicator.EWMA(data, window)
        data['EWMA_SMA_ratio{}'.format(window)] = data['EWMA_{}'.format(window)] / data['close_ma{}'.format(window)]
        # ROC
        data = indicator.ROC(data, window)
        # forceindex
        data = indicator.ForceIndex(data, window)
        data['FI_OBV_ratio{}'.format(window)] = data['FI_{}'.format(window)] / data['OBV']

    data.dropna(inplace=True)

    return data

def get_train_data(chart_data):
    TRAIN_DATA_COLUMNS = []
    for col in chart_data.columns:
        if 'ratio' in col:
            TRAIN_DATA_COLUMNS.append(col)
            
    TRAIN_DATA = chart_data[TRAIN_DATA_COLUMNS]

    return TRAIN_DATA


if __name__ == "__main__":
    ticker = 'MSFT'
    start_date = '2010-01-01'

    df = load_data(ticker, start_date)
    df['Date'] = df['Date'].astype('str')

    predf = preprocess(df)
    print(predf.head())




