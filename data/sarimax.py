from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split

import alpaca_trade_api as tradeapi
from datetime import datetime
import pytz

api_key = 'AKQB26E5HLHLA54FOT9T'
api_secret = 'hNgAGISeVTThSFOoa1biaGcRuWYD8HvOvtb1AB7c'
base_url = 'https://paper-api.alpaca.markets'  
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

symbol = 'AAPL' 
start_date = datetime(2000, 1, 1).astimezone(pytz.timezone('UTC')) 
end_date = datetime(2022, 1, 31).astimezone(pytz.timezone('UTC'))  

daily_prices = api.get_bars(symbol, tradeapi.TimeFrame.Day, start_date.isoformat(), end_date.isoformat()).df
print(daily_prices.columns)

required_columns = ['close', 'high', 'low', 'trade_count', 'open', 'volume']
for col in required_columns:
    if col not in daily_prices.columns:
        raise ValueError(f"Missing required column: {col}")

X = daily_prices[['close', 'high', 'low', 'trade_count', 'open']]
Y = daily_prices['volume']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

arima_model = ARIMA(y_train, order=(5, 1, 0))  # (p, d, q) order
arima_result = arima_model.fit()

arima_forecast = arima_result.forecast(steps=len(y_test))

mse = mean_squared_error(y_test, arima_forecast)
r2 = r2_score(y_test, arima_forecast)
print(f"ARIMA Model - Mean Squared Error: {sqrt(mse)}")
print(f"ARIMA Model - R-squared: {r2}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Volume')
plt.plot(y_test.index, arima_forecast, label='ARIMA Forecast', color='red')
plt.legend()
plt.show()

# fit SARIMAX model on training data
sarimax_model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_result = sarimax_model.fit()

# forecast on test data
sarimax_forecast = sarimax_result.get_forecast(steps=len(y_test), exog=X_test)
sarimax_pred = sarimax_forecast.predicted_mean

mse = mean_squared_error(y_test, sarimax_pred)
r2 = r2_score(y_test, sarimax_pred)
print(f"SARIMAX Model - Mean Squared Error: {sqrt(mse)}")
print(f"SARIMAX Model - R-squared: {r2}")
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Volume')
plt.plot(y_test.index, sarimax_pred, label='SARIMAX Forecast', color='red')
plt.legend()
plt.show()

