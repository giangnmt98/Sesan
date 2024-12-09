import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# 1. Đọc dữ liệu từ file CSV
def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV, đảm bảo định dạng đúng.
    Args:
        file_path (str): Đường dẫn tới file CSV
    Returns:
        pd.DataFrame: Dữ liệu với cột Date làm chỉ số
    """
    data = pd.read_csv(file_path)
    data.columns = ['Date', 'Value']  # Đặt tên cột
    data['Date'] = pd.to_datetime(data['Date'])
    return data.set_index('Date')


# 2. Chia dữ liệu thành tập train và test
def split_data(data, split_ratio=0.8, split_date=None):
    """
    Chia dữ liệu thành tập train và test dựa trên tỷ lệ hoặc ngày cụ thể.
    Args:
        data (pd.DataFrame): Dữ liệu chuỗi thời gian
        split_ratio (float): Tỷ lệ dữ liệu train (mặc định 80%)
        split_date (str): Ngày phân chia định dạng 'YYYY-MM-DD' (nếu có)
    Returns:
        pd.DataFrame, pd.DataFrame: Tập train và test
    """
    if split_date:
        split_point = pd.to_datetime(split_date)
        train = data[data.index < split_point]
        test = data[data.index >= split_point]
    else:
        split_index = int(len(data) * split_ratio)
        train = data[:split_index]
        test = data[split_index:]
    return train, test


# 3. Vẽ biểu đồ train/test
def plot_train_test(train, test):
    """
    Vẽ biểu đồ hiển thị tập train và test bằng Plotly.
    Args:
        train (pd.DataFrame): Tập dữ liệu train
        test (pd.DataFrame): Tập dữ liệu test
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Value'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test['Value'], mode='lines', name='Test'))
    fig.update_layout(title='Train/Test Split',
                      xaxis_title='Date',
                      yaxis_title='Values',
                      legend_title='Legend')
    fig.show()


# 4. Dự báo bằng ARIMA
def arima_forecasting(train, test, order=(5, 1, 0)):
    """
    Dự báo dữ liệu sử dụng mô hình ARIMA.
    Args:
        train (pd.DataFrame): Tập dữ liệu train
        test (pd.DataFrame): Tập dữ liệu test
        order (tuple): Tham số ARIMA (mặc định là (5, 1, 0))
    Returns:
        np.ndarray: Giá trị dự báo của ARIMA
    """
    model = ARIMA(train, order=order)
    result = model.fit()
    return result.forecast(steps=len(test))


# 5. Dự báo bằng Prophet
def prophet_forecasting(train, test):
    """
    Dự báo dữ liệu sử dụng mô hình Prophet.
    Args:
        train (pd.DataFrame): Tập dữ liệu train
        test (pd.DataFrame): Tập dữ liệu test
    Returns:
        pd.Series: Giá trị dự báo của Prophet
    """
    prophet_data = train.reset_index().rename(columns={"Date": "ds", "Value": "y"})
    model = Prophet( daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
    changepoint_prior_scale=0.5)
    model.fit(prophet_data)
    future = test.reset_index().rename(columns={"Date": "ds"})
    forecast = model.predict(future)
    return forecast['yhat']


# 6. Dự báo bằng Autoencoder (PyTorch)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def autoencoder_forecasting(train, test, encoding_dim=100, epochs=100, batch_size=16, learning_rate=0.0001):
    """
    Dự báo dữ liệu sử dụng Autoencoder.
    Args:
        train (pd.DataFrame): Tập dữ liệu train
        test (pd.DataFrame): Tập dữ liệu test
        encoding_dim (int): Số chiều của lớp mã hóa
        epochs (int): Số epoch huấn luyện
        batch_size (int): Kích thước batch
        learning_rate (float): Tốc độ học
    Returns:
        np.ndarray: Giá trị dự báo của Autoencoder
    """
    # Chuẩn hóa dữ liệu
    train_values = train['Value'].values
    test_values = test['Value'].values
    scaler = (train_values.min(), train_values.max())
    train_scaled = (train_values - scaler[0]) / (scaler[1] - scaler[0])
    test_scaled = (test_values - scaler[0]) / (scaler[1] - scaler[0])

    # Sử dụng kích thước ngắn nhất cho Autoencoder
    input_dim = min(len(train_scaled), len(test_scaled))
    train_scaled = train_scaled[:input_dim]
    test_scaled = test_scaled[:input_dim]

    # Chuyển dữ liệu sang Tensor
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32).unsqueeze(0)

    # Khởi tạo Autoencoder
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Huấn luyện Autoencoder
    train_losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # Vẽ biểu đồ quá trình huấn luyện
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(epochs), train_losses, label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Autoencoder Training Loss')
    # plt.legend()
    # plt.show()

    # Dự báo
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        reconstructed = model(test_tensor).squeeze(0).numpy()

    forecast = reconstructed * (scaler[1] - scaler[0]) + scaler[0]
    return forecast

# 7. Đánh giá kết quả
def evaluate_forecast(actual, predicted):
    """
    Đánh giá kết quả dự báo bằng MAE và RMSE.
    Args:
        actual (np.ndarray): Giá trị thực tế
        predicted (np.ndarray): Giá trị dự báo
    Returns:
        float, float: MAE và RMSE
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


# 8. Vẽ biểu đồ so sánh dự báo
def plot_forecast_comparison(test, forecasts):
    """
    Vẽ biểu đồ so sánh kết quả dự báo từ nhiều mô hình với dữ liệu thực tế bằng Plotly.
    Args:
        test (pd.DataFrame): Tập dữ liệu test
        forecasts (dict): Từ điển chứa các dự báo từ các mô hình, dạng {"Model Name": forecast_values}
    """
    fig = go.Figure()
    # Thêm dữ liệu thực tế
    fig.add_trace(go.Scatter(x=test.index, y=test['Value'], mode='lines', name='Actual'))
    # Thêm các dự báo từ các mô hình
    for model_name, forecast_values in forecasts.items():
        fig.add_trace(go.Scatter(x=test.index, y=forecast_values, mode='lines', name=model_name))
    fig.update_layout(title='Forecast Comparison with Actual Data',
                      xaxis_title='Date',
                      yaxis_title='Values',
                      legend_title='Models')
    fig.show()


# 9. Quy trình chính
def main(file_path, split_date=None):
    """
    Quy trình chính: Tải dữ liệu, chia train/test, dự báo và đánh giá.
    Args:
        file_path (str): Đường dẫn tới file CSV
        split_date (str): Ngày phân chia định dạng 'YYYY-MM-DD' (nếu có)
    """
    # Tải dữ liệu
    data = load_data(file_path)

    # Chia dữ liệu
    train, test = split_data(data, split_date=split_date)

    # Vẽ tập train/test
    # plot_train_test(train, test)

    # Dự báo với ARIMA
    arima_pred = arima_forecasting(train, test)

    # Dự báo với Prophet
    prophet_pred = prophet_forecasting(train, test)

    # Dự báo với Autoencoder
    autoencoder_pred = autoencoder_forecasting(train, test)

    # Đánh giá kết quả
    arima_mae, arima_rmse = evaluate_forecast(test['Value'].values, arima_pred)
    prophet_mae, prophet_rmse = evaluate_forecast(test['Value'].values, prophet_pred)
    autoencoder_mae, autoencoder_rmse = evaluate_forecast(test['Value'].values, autoencoder_pred)

    print(f"ARIMA - MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
    print(f"Prophet - MAE: {prophet_mae:.2f}, RMSE: {prophet_rmse:.2f}")
    print(f"Autoencoder - MAE: {autoencoder_mae:.2f}, RMSE: {autoencoder_rmse:.2f}")

    # Vẽ biểu đồ dự báo
    # forecasts = {
    #     "ARIMA Forecast": arima_pred,
    #     "Prophet Forecast": prophet_pred.values,
    #     "Autoencoder Forecast": autoencoder_pred
    # }
    # plot_forecast_comparison(test,{"ARIMA Forecast": arima_pred})
    plot_forecast_comparison(test,{"Prophet Forecast": prophet_pred.values})
    # plot_forecast_comparison(test,{"Autoencoder Forecast": autoencoder_pred})

# Chạy chương trình
if __name__ == "__main__":
    file_path = "data/data_sample_for_forcasting.csv"  # Thay bằng đường dẫn file CSV của bạn
    main(file_path)
