import pandas as pd
import numpy as np
import plotly.graph_objects as go

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = ['Date', 'Value']
    data['Date'] = pd.to_datetime(data['Date'])
    return data.set_index('Date')

def split_data(data, split_ratio=0.8, split_date=None):
    if split_date:
        split_point = pd.to_datetime(split_date)
        train = data[data.index < split_point]
        test = data[data.index >= split_point]
    else:
        split_index = int(len(data) * split_ratio)
        train = data[:split_index]
        test = data[split_index:]
    return train, test

def plot_train_test(train, test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Value'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test['Value'], mode='lines', name='Test'))
    fig.update_layout(title='Train/Test Split',
                      xaxis_title='Date',
                      yaxis_title='Values',
                      legend_title='Legend')
    fig.show()


# Generate synthetic time series data
def generate_data(n_samples=2000, amplitude=1, frequency=1, noise_level=0.1):
    """
    Generate synthetic time series data resembling a horizontal sine wave with added noise.
    Args:
        n_samples (int): Number of samples in the time series
        amplitude (float): Amplitude of the sine wave
        frequency (float): Frequency of the sine wave
        noise_level (float): Standard deviation of Gaussian noise
    Returns:
        pd.DataFrame: A DataFrame with 'Date' and 'Value' columns
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='10T')  # 10-minute intervals
    x = np.linspace(0, 2 * np.pi * frequency * n_samples / 100, n_samples)
    values = amplitude * np.sin(x) + np.random.normal(scale=noise_level, size=n_samples)
    return pd.DataFrame({'Date': dates, 'Value': values})

