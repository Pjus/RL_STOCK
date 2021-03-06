from data_manager import load_data, preprocess, get_train_data
from environment import Environment


ticker = 'MSFT'
start_date = '2010-01-01'

df = load_data(ticker, start_date)
df['Date'] = df['Date'].astype('str')
# print(df.head())
predf = preprocess(df)
# print(predf.head())

print(Environment(predf).observe())
print(Environment(predf).get_price())
