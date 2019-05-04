import pandas as pd 
import quandl, math
import numpy as np 
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  

# Linear Regression
df = pd.read_csv('wiki_prices_googl.csv')
df = df[df['ticker'].str.match('GOOGL')]
df['date'] = pd.to_datetime(df['date'])

df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0

df = df[['adj_close', 'HL_PCT', 'PCT_change', 'adj_volume']]

forecast_col = 'adj_close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# Questions around scaling with real data
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(forecast_out)
print(accuracy*100)