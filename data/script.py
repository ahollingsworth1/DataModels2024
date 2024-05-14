from math import sqrt
import alpaca_trade_api as tradeapi
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score


# API credentials
api_key = 'AKQB26E5HLHLA54FOT9T'
api_secret = 'hNgAGISeVTThSFOoa1biaGcRuWYD8HvOvtb1AB7c'
base_url = 'https://paper-api.alpaca.markets'  

# Initialize the Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

symbol = 'SPY' 
start_date = datetime(2021, 1, 1).astimezone(pytz.timezone('UTC')) 
end_date = datetime(2024, 4, 15).astimezone(pytz.timezone('UTC'))  

daily_prices = api.get_bars(symbol, tradeapi.TimeFrame.Day, start_date.isoformat(), end_date.isoformat()).df
print(daily_prices)


X = daily_prices[['close', 'high', 'low', 'trade_count', 'open', 'vwap']]

Y = daily_prices['volume']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

print(X_train.shape[1])

# NN model
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(10, activation='relu'),
    Dense(1) 
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)  

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions.flatten())
r2 = r2_score(y_test, predictions.flatten())

print(f"Mean Squared Error: {sqrt(mse)}")
print(f"R-squared: {r2}")



