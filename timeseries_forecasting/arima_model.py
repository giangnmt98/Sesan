from statsmodels.tsa.arima.model import ARIMA

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
    # Ensure train is a 1D numeric array
    train_cleaned = train['Value'].dropna().astype(float)

    # Fit the ARIMA model
    model = ARIMA(train_cleaned, order=order)
    result = model.fit()

    # Forecast for the test period
    forecast = result.forecast(steps=len(test))

    return forecast
