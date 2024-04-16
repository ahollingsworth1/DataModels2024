
""" 

link: https://app.alpaca.markets/account/login
Username: ryanflatley0@gmail.com
Password: 2022wuwu5RF!

 """


import alpaca_trade_api as tradeapi
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# API credentials
api_key = 'AKQB26E5HLHLA54FOT9T'
api_secret = 'hNgAGISeVTThSFOoa1biaGcRuWYD8HvOvtb1AB7c'
base_url = 'https://paper-api.alpaca.markets'  

# Initialize the Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

symbol = 'AAPL' 
start_date = datetime(2000, 1, 1).astimezone(pytz.timezone('UTC')) 
end_date = datetime(2022, 1, 31).astimezone(pytz.timezone('UTC'))  

daily_prices = api.get_bars(symbol, tradeapi.TimeFrame.Day, start_date.isoformat(), end_date.isoformat()).df

print(daily_prices)

X = daily_prices[['close', 'high', 'low', 'trade_count', 'open']]
Y = daily_prices['volume']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# i initially had this a decision tree but the metrics were bad so switched it to forest which improved it some
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")