import pandas as pd
from prophet import Prophet

def prophet_forecasting(train, test):
    prophet_data = train.reset_index().rename(columns={"Date": "ds", "Value": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.01)
    model.fit(prophet_data)
    future = test.reset_index().rename(columns={"Date": "ds"})
    forecast = model.predict(future)
    return forecast['yhat']
