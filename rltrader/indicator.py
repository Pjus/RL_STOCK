import pandas as pd
import numpy as np

def DMI(data, n=None, n_ADX=None):
    i = 0
    UpI = [0]
    DoI = [0]

    while i + 1 <= data.index[-1] :
        UpMove = data.loc[i + 1, "High"] - data.loc[i, "High"]
        DoMove = data.loc[i, "Low"] - data.loc[i+1, "Low"]
        if UpMove > DoMove and UpMove > 0 :
            UpD = UpMove
        else :
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0 :
            DoD = DoMove
        else :
            DoD = 0
        DoI.append(DoD)
        i = i + 1

    i = 0
    TR_l = [0]
    while i < data.index[-1]:
        TR = max(data.loc[i + 1, 'High'], data.loc[i, 'Close']) - min(data.loc[i + 1, 'Low'], data.loc[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=1).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=1).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=1).mean() / ATR)
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=1).mean(),
                    name='ADX_' + str(n) + '_' + str(n_ADX))
                    
    data["PDI_{}".format(n)],data["MDI_{}".format(n)],data["ADX_{}".format(n_ADX)] = PosDI, NegDI, ADX
    
    return data
    

"""
중심선: n기간 동안의 이동평균(SMA)

상단선: 중심선 + Kσ(일반적으로 K는 2배를 많이 사용함)

하단선: 중심선 - Kσ(일반적으로 K는 2배를 많이 사용함)

"""

def fnBolingerBand(m_DF, n=None, k=2):
    m_DF['Bol_upper_{}'.format(n)] = m_DF['Close'].rolling(window=n).mean() + k * m_DF['Close'].rolling(window=n).std()
    m_DF['Bol_lower_{}'.format(n)] = m_DF['Close'].rolling(window=n).mean() - k * m_DF['Close'].rolling(window=n).std()
    return m_DF



def faster_OBV(data):
    close, volume = data['Close'], data['Volume']
    # obv 값이 저장될 리스트를 생성합니다.
    obv_value = [None] * len(close)
    obv_value[0] = volume.iloc[0]
    # 마지막에 사용할 인덱스를 저장해 둡니다.
    index = close.index

    # 연산에서 사용하기 위해 리스트 형태로 바꾸어 줍니다.
    close = list(close)
    volume = list(volume)
    
    # OBV 산출공식을 구현
    for i in range(1,len(close)):
    
        if close[i] > close[i-1] : 
            obv_value[i] = obv_value[i-1] + volume[i]
            
        elif close[i] < close[i-1] :
            obv_value[i] = obv_value[i-1] - volume[i]
            
        else:
            obv_value[i] = obv_value[i-1]
            
    # 계산된 리스트 결과물을 마지막에 Series 구조로 변환해 줍니다.
    obv = pd.Series(obv_value, index=index)
    data['OBV'] = obv
    return data

# 30이하면 과매도, 70 이상이면 과매수
def fnRSI(df, m_N=7):
    delta = df['Close'].diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = pd.DataFrame(dUp).rolling(window=14).mean()
    RolDown = pd.DataFrame(dDown).rolling(window=14).mean().abs()

    RS = RolUp / RolDown
    RSI = RS / (1+RS)
    RSI_MACD = pd.DataFrame(RSI).rolling(window=6).mean()
    df['RSI_MACD_{}'.format(m_N)] = RSI_MACD

    return df



# Commodity Channel Index 
def CCI(data, ndays): 
    TP = (data['High'] + data['Low'] + data['Close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
                    name = 'CCI') 
    data['CCI_{}'.format(ndays)] = CCI / 100
    return data

# Ease Of Movement (EVM) Code

# Load the necessary packages and modules
 
# Ease of Movement 
def EVM(data, ndays): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br 
    EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name = 'EVM') 
    data['EVM_{}'.format(ndays)] = EVM_MA
    return data 

# Simple Moving Average 
def SMA(data, ndays): 
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name = 'SMA') 
    data['SMA_{}'.format(ndays)] = SMA
    return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                    name = 'EWMA_' + str(ndays)) 
    data['EWMA_{}'.format(ndays)] = EMA
    return data

# Rate of Change (ROC)
def ROC(data,n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D,name='Rate of Change')
    data['ROC_{}'.format(n)] = ROC
    return data 

# Force Index 
def ForceIndex(data, ndays): 
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    data['FI_{}'.format(ndays)] = FI
    return data