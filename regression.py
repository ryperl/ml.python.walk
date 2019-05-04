import pandas as pd 
import quandl, math, datetime, time
import numpy as np 
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle

style.use('ggplot')

# Linear Regression
df = pd.read_csv('wiki_prices_googl.csv')
df = df.set_index('date')

df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0

df = df[['adj_close', 'HL_PCT', 'PCT_change', 'adj_volume']]

forecast_col = 'adj_close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan 

last_date = df.iloc[-1].name
tmp_date = pd.to_datetime(last_date)
ts = tmp_date.timestamp()
last_unix = ts

one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]

print(df.head())
print(df.tail())

df['adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()