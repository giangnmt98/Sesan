import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

def plot_forecast_comparison(test, forecasts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test['Value'], mode='lines', name='Actual'))
    for model_name, forecast_values in forecasts.items():
        fig.add_trace(go.Scatter(x=test.index, y=forecast_values, mode='lines', name=model_name))
    fig.update_layout(title='Forecast Comparison with Actual Data',
                      xaxis_title='Date',
                      yaxis_title='Values',
                      legend_title='Models')
    fig.show()